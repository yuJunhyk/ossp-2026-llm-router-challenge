# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""선형(dual ridge) 학습 라우터 — 표준 라이브러리 전용 추론.

학습(analysis/train_linear.py)은 NumPy를 쓰지만, 이 런타임은 아티팩트
JSON(모델별 score·log-비용 선형 계수 + smear/σ 보정 + tier별 β·margin)만
읽어 표준 라이브러리로 추론한다. 배분은 learned_policy.allocate(볼록껍질
그리디)로 tier 배치 전체를 한 번에 결정한다. 문항별 결정은 프롬프트 내용과
tier에만 의존하며, episode_id·순서·split을 읽지 않는다.

예측기·β·margin은 Dev를 쓰지 않고 train 그룹-fold CV로만 선택했다
(analysis/rematch.py, build/rematch-report.json).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .learned_features import TOTAL_DIM, extract_sparse
from .learned_policy import allocate
from .heuristic import episode_text
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)

ARTIFACT_TYPE = "ossp-learned-linear-router-v1"
ARTIFACT_RESOURCE = "learned-router.v1.json"


@dataclass(frozen=True)
class LinearArtifact:
    score_weights: Dict[str, Tuple[float, ...]]
    logcost_weights: Dict[str, Tuple[float, ...]]
    cost_smear: Dict[str, float]
    cost_sigma: Dict[str, float]
    tier_config: Dict[str, Tuple[float, float]]  # tier -> (beta, margin)
    policy_id: str
    policy_digest: str


def _weight_row(value: Any, name: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != TOTAL_DIM:
        raise ProtocolError(f"{name} 계수 길이가 {TOTAL_DIM}이 아닙니다.")
    row = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in row):
        raise ProtocolError(f"{name} 계수에 유한하지 않은 값이 있습니다.")
    return row


def parse_artifact(value: Any) -> LinearArtifact:
    """구조를 엄격히 검증한다 — 길이·범위 불일치는 조용한 오예측 대신 오류."""
    if not isinstance(value, dict) or value.get("artifact_type") != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 learned artifact입니다.")
    if value.get("schema_version") != 1:
        raise ProtocolError("지원하지 않는 learned artifact 버전입니다.")
    if list(value.get("models", ())) != list(MODEL_IDS):
        raise ProtocolError("artifact의 모델 목록이 정책과 다릅니다.")
    weights = value["weights"]
    score_weights = {
        m: _weight_row(weights["score"][m], f"score[{m}]") for m in MODEL_IDS
    }
    logcost_weights = {
        m: _weight_row(weights["log_cost"][m], f"log_cost[{m}]") for m in MODEL_IDS
    }
    cost_smear, cost_sigma = {}, {}
    for m in MODEL_IDS:
        smear = float(value["cost_smear"][m])
        sigma = float(value["cost_sigma"][m])
        if not (math.isfinite(smear) and smear > 0):
            raise ProtocolError("cost_smear는 양의 유한값이어야 합니다.")
        if not (math.isfinite(sigma) and sigma >= 0):
            raise ProtocolError("cost_sigma는 0 이상의 유한값이어야 합니다.")
        cost_smear[m] = smear
        cost_sigma[m] = sigma
    tier_config = {}
    raw_config = value["tier_config"]
    if set(raw_config) != set(TIERS):
        raise ProtocolError("tier_config가 완전하지 않습니다.")
    for tier in TIERS:
        beta = float(raw_config[tier]["beta"])
        margin = float(raw_config[tier]["margin"])
        if not (0 <= beta <= 8):
            raise ProtocolError("beta는 0 이상 8 이하여야 합니다.")
        if not (0 < margin <= 1):
            raise ProtocolError("margin은 0보다 크고 1 이하여야 합니다.")
        tier_config[tier] = (beta, margin)
    return LinearArtifact(
        score_weights=score_weights,
        logcost_weights=logcost_weights,
        cost_smear=cost_smear,
        cost_sigma=cost_sigma,
        tier_config=tier_config,
        policy_id=str(value["policy_id"]),
        policy_digest=str(value["policy_sha256"]),
    )


def load_bundled_artifact() -> Optional[LinearArtifact]:
    """패키지에 동봉된 아티팩트를 읽는다. 파일이 없으면 None."""
    try:
        payload = (
            importlib_resources.files("ossp_router.resources")
            .joinpath(ARTIFACT_RESOURCE)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    return parse_artifact(json.loads(payload))


def load_artifact(path: Path) -> LinearArtifact:
    return parse_artifact(json.loads(path.read_text(encoding="utf-8")))


def _dot(features: Dict[int, float], weights: Tuple[float, ...]) -> float:
    return math.fsum(weights[index] * value for index, value in features.items())


def predict_episode(
    episode: Episode, artifact: LinearArtifact
) -> Dict[str, Tuple[float, float]]:
    """모델별 (기대 점수, smear 보정 기대 비용). β 비관은 tier 단계에서 적용."""
    features = extract_sparse(episode_text(episode))
    prediction: Dict[str, Tuple[float, float]] = {}
    for model in MODEL_IDS:
        score = min(1.0, max(0.0, _dot(features, artifact.score_weights[model])))
        log_cost = min(5.0, max(-20.0, _dot(features, artifact.logcost_weights[model])))
        cost = math.exp(log_cost) * artifact.cost_smear[model]
        prediction[model] = (score, cost)
    return prediction


def make_learned_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: LinearArtifact,
    tier: str,
) -> Submission:
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if artifact.policy_id != policy.policy_id:
        raise ProtocolError("artifact와 정책의 policy_id가 다릅니다.")
    if artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("artifact와 현재 정책의 SHA-256이 다릅니다.")
    beta, margin = artifact.tier_config[tier]
    pessimism = {
        model: (
            1.0
            if model == MODEL_IDS[0]
            else math.exp(beta * artifact.cost_sigma[model])
        )
        for model in MODEL_IDS
    }
    predictions = []
    for episode in inputs.episodes:
        base = predict_episode(episode, artifact)
        predictions.append(
            {
                model: (score, cost * pessimism[model])
                for model, (score, cost) in base.items()
            }
        )
    selected = allocate(
        predictions, float(policy.tiers[tier].budget_multiplier), margin
    )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return parse_submission(submission_to_dict(submission))
