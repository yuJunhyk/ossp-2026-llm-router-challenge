# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""ridge + GBM(uplift 직접 학습) 앙상블 라우터 — 표준 라이브러리 전용 추론.

학습(analysis/train_ens.py)은 NumPy·LightGBM을 쓰지만, 이 런타임은 아티팩트
JSON(선형 head 계수 + 평탄화된 GBM 트리)만 읽어 표준 라이브러리로 추론한다.
문항별 결정은 프롬프트 내용과 tier에만 의존하며, episode_id·순서·split을
읽지 않는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .heuristic import episode_text, extract_features
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

ARTIFACT_TYPE = "ossp-ens-router-v1"
ARTIFACT_RESOURCE = "ens-artifact.v1.json"

# --- 특징 추출 (baselines/hash_regex.py와 동일 정의를 런타임에 내장) ---
import re

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_FORMAL_REASONING = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM_ANALYSIS = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_MULTI_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|" r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)


def _stable_hash(value: str) -> int:
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def raw_feature_vector(episode: Episode, hash_bins: int) -> Tuple[float, ...]:
    features = extract_features(episode)
    text = episode_text(episode)
    dense = (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.sentence_count),
        math.log1p(features.message_count),
        features.hangul_ratio,
        math.log1p(features.code_marker_count),
        math.log1p(features.math_marker_count),
        features.numeric_density,
        float(features.long_context),
        math.log1p(features.reasoning_marker_count),
        float(bool(_FORMAL_REASONING.search(text))),
        float(bool(_PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(_MULTI_CONSTRAINT.findall(text))),
        float(bool(_SIMPLE_TRANSFORM.search(text))),
    )
    bins = [0.0] * hash_bins
    tokens = []
    for token in _TOKEN.findall(text):
        normalized = token.casefold()
        if normalized.isdecimal():
            normalized = "<number>"
        tokens.append(normalized)
    hashed = [f"w1:{token}" for token in tokens]
    hashed.extend(f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:]))
    for value in hashed:
        digest = _stable_hash(value)
        index = digest & (hash_bins - 1)
        bins[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return dense + tuple(bins)


# --- 아티팩트 ---


@dataclass(frozen=True)
class EnsArtifact:
    hash_bins: int
    gbm_weight: float
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    ridge_intercept: Tuple[float, ...]
    ridge_coefficients: Tuple[Tuple[float, ...], ...]  # [target][feature]
    gbm_trees: Any  # [target][seed][tree] 평탄 구조
    tier_safety_ratios: Mapping[str, float]
    policy_id: str
    policy_digest: str


def parse_artifact(value: Any) -> EnsArtifact:
    if not isinstance(value, dict) or value.get("artifact_type") != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 ens artifact입니다.")
    if value.get("schema_version") != 1:
        raise ProtocolError("지원하지 않는 ens artifact 버전입니다.")
    hash_bins = value["hash_bins"]
    if not isinstance(hash_bins, int) or hash_bins & (hash_bins - 1) or hash_bins <= 0:
        raise ProtocolError("hash_bins는 2의 거듭제곱이어야 합니다.")
    ridge = value["ridge"]
    length = len(ridge["feature_mean"])
    safety = value["tier_safety_ratios"]
    if set(safety) != set(TIERS):
        raise ProtocolError("tier_safety_ratios가 완전하지 않습니다.")
    for tier, ratio in safety.items():
        if not 0 < float(ratio) <= 1:
            raise ProtocolError("안전계수는 0보다 크고 1 이하여야 합니다.")
    return EnsArtifact(
        hash_bins=hash_bins,
        gbm_weight=float(value["ensemble_gbm_weight"]),
        feature_mean=tuple(float(v) for v in ridge["feature_mean"]),
        feature_scale=tuple(float(v) for v in ridge["feature_scale"]),
        ridge_intercept=tuple(float(v) for v in ridge["intercept"]),
        ridge_coefficients=tuple(
            tuple(float(v) for v in row) for row in ridge["coefficients"]
        ),
        gbm_trees=value["gbm"]["trees"],
        tier_safety_ratios={t: float(safety[t]) for t in TIERS},
        policy_id=str(value["policy_id"]),
        policy_digest=str(value["policy_sha256"]),
    )


def load_bundled_artifact() -> Optional[EnsArtifact]:
    """패키지에 동봉된 아티팩트를 읽는다. 없으면 None."""
    try:
        payload = (
            importlib_resources.files("ossp_router.resources")
            .joinpath(ARTIFACT_RESOURCE)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    return parse_artifact(json.loads(payload))


def load_artifact(path: Path) -> EnsArtifact:
    return parse_artifact(json.loads(path.read_text(encoding="utf-8")))


# --- 추론 ---


def _eval_trees(trees: Sequence[Mapping[str, Any]], features: Sequence[float]) -> float:
    total = 0.0
    for tree in trees:
        feats = tree["f"]
        if not feats:
            total += tree["v"][0]
            continue
        thresholds = tree["t"]
        lefts = tree["l"]
        rights = tree["r"]
        node = 0
        while node >= 0:
            if features[feats[node]] <= thresholds[node]:
                node = lefts[node]
            else:
                node = rights[node]
        total += tree["v"][-node - 1]
    return total


def predict_episode(
    episode: Episode, artifact: EnsArtifact
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    raw = raw_feature_vector(episode, artifact.hash_bins)
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            raw, artifact.feature_mean, artifact.feature_scale
        )
    )
    # ridge: 표준화 특징의 선형 결합 (레벨 타깃 6개)
    ridge_pred = [
        intercept + math.fsum(c * v for c, v in zip(coeffs, standardized))
        for intercept, coeffs in zip(
            artifact.ridge_intercept, artifact.ridge_coefficients
        )
    ]
    # GBM: 원시 특징으로 diff 타깃 예측 후 레벨로 재구성, 시드 평균
    gbm_diff = []
    for per_seed in artifact.gbm_trees:
        value = math.fsum(_eval_trees(trees, raw) for trees in per_seed) / len(per_seed)
        gbm_diff.append(value)
    gbm_pred = [
        gbm_diff[0],
        gbm_diff[0] + gbm_diff[1],
        gbm_diff[0] + gbm_diff[2],
        gbm_diff[3],
        gbm_diff[4],
        gbm_diff[5],
    ]
    w = artifact.gbm_weight
    ens = [(1 - w) * r + w * g for r, g in zip(ridge_pred, gbm_pred)]
    scores = {
        model_id: min(1.0, max(0.0, ens[index]))
        for index, model_id in enumerate(MODEL_IDS)
    }
    costs = {
        model_id: math.exp(min(50.0, max(-50.0, ens[3 + index])))
        for index, model_id in enumerate(MODEL_IDS)
    }
    light = costs[MODEL_IDS[0]]
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], light * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
    return scores, costs


# --- 선택 (baselines/hash_regex.py의 라그랑주 λ 이분 탐색과 동일 로직) ---


def select_models(
    predicted_scores: Sequence[Mapping[str, float]],
    predicted_costs: Sequence[Mapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    if len(predicted_scores) != len(predicted_costs) or not predicted_scores:
        raise ValueError(
            "예측 score와 cost는 같은 길이의 비어 있지 않은 배열이어야 합니다."
        )
    light_total = math.fsum(row[MODEL_IDS[0]] for row in predicted_costs)
    effective_ratio = max(1.0, budget_multiplier * safety_ratio)
    cap = light_total * effective_ratio

    def choose(penalty: float) -> Tuple[Tuple[str, ...], float]:
        selected = []
        for scores, costs in zip(predicted_scores, predicted_costs):
            model_id = max(
                MODEL_IDS,
                key=lambda candidate: (
                    scores[candidate] - penalty * costs[candidate] / light_total,
                    -MODEL_IDS.index(candidate),
                ),
            )
            selected.append(model_id)
        total = math.fsum(
            costs[model_id] for costs, model_id in zip(predicted_costs, selected)
        )
        return tuple(selected), total

    selected, total = choose(0.0)
    if total > cap:
        low = 0.0
        high = 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low = high
            high *= 2.0
            selected, total = choose(high)
        for _iteration in range(80):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high = middle
                selected, total = candidate, candidate_total
            else:
                low = middle
    if total > cap:
        selected = tuple(MODEL_IDS[0] for _row in predicted_scores)
        total = light_total
    return selected, total / light_total


def make_ens_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: EnsArtifact,
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
    predictions = [predict_episode(episode, artifact) for episode in inputs.episodes]
    scores = [item[0] for item in predictions]
    costs = [item[1] for item in predictions]
    selected, _ratio = select_models(
        scores,
        costs,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=artifact.tier_safety_ratios[tier],
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
