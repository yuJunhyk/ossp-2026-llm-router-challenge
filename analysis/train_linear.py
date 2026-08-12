# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""선형(dual ridge) 라우터 최종 트레이너 — learned-router.v1.json 생성.

선택 근거 (analysis/rematch.py, Dev 미사용):
- 템플릿 그룹 5-fold × 3 seed = 15 fold 재대결에서 linear(λ=10)가
  weighted CV 0.6553으로 1위 (ens 최고 0.6540, v1.2 구성 w=0.5는 0.6465).
- tier별 (β, margin)은 같은 재대결의 제약 캘리브레이션 승자를 그대로 쓴다:
  15 fold 전부 실현 예산비 ≤ 한도×{0.90,0.95,0.95} + 유형 편향 부분집합
  9종 ≤ 한도×0.98.
- 순수 stdlib 추론(레이턴시 이점)과 dev-clean 계보가 근소 우위의 동률
  판정 기준.

여기서는 전체 train으로 최종 계수를 적합하고, 런타임(learned_router)이
아티팩트를 읽어 동일 예측을 내는지 표본 대조까지 수행한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np

from rematch import cost_stats, episode_text, fit_dual_ridge
from os2_features import TOTAL_DIM, extract_sparse
from os2_policy import allocate
from ossp_router import learned_router
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)

LAMBDA = 10.0
# analysis/rematch.py 승자 설정 (build/rematch-report.json)
TIER_CONFIG = {
    "fast": {"beta": 0.5, "margin": 0.90},
    "balanced": {"beta": 0.5, "margin": 0.96},
    "premium": {"beta": 1.0, "margin": 0.80},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="선형 라우터 아티팩트 학습")
    parser.add_argument(
        "--input", type=Path, default=REPO / "data/materialized/train/inputs.json"
    )
    parser.add_argument(
        "--outcomes", type=Path, default=REPO / "data/train/outcomes.json"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO / "src/ossp_router/resources/learned-router.v1.json",
    )
    parser.add_argument(
        "--report", type=Path, default=REPO / "build/train-linear-report.json"
    )
    args = parser.parse_args()

    policy = load_bundled_policy()
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    rates = {m: policy.models[m] for m in MODEL_IDS}
    truth = {}
    for oc in outcomes.outcomes:
        r = rates[oc.model_id]
        cost = (
            float(r.fixed_cost)
            + (
                oc.input_tokens * float(r.input_token_rate)
                + oc.output_tokens * float(r.output_token_rate)
            )
            / policy.token_unit
        )
        truth[(oc.episode_id, oc.model_id)] = (float(oc.score), cost)
    episode_ids = [ep.episode_id for ep in inputs.episodes]

    print(f"[1/3] 특징 행렬 구축 (n={n}, dim={TOTAL_DIM})")
    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val
    Y = np.zeros((n, 6))
    for i, eid in enumerate(episode_ids):
        for j, m in enumerate(MODEL_IDS):
            score, cost = truth[(eid, m)]
            Y[i, j] = score
            Y[i, 3 + j] = math.log(max(cost, 1e-9))

    print(f"[2/3] 전체 train 적합 (λ={LAMBDA}) + 아티팩트 생성")
    W = fit_dual_ridge(X, Y, LAMBDA)
    smear, sigma = cost_stats((X @ W)[:, 3:], Y[:, 3:])

    artifact_payload = {
        "artifact_type": learned_router.ARTIFACT_TYPE,
        "schema_version": 1,
        "trained_on": (
            "public train split only (1,760 episodes); predictor/config selected "
            "by template-group 5-fold x 3-seed CV (analysis/rematch.py), dev untouched"
        ),
        "models": list(MODEL_IDS),
        "lambda": LAMBDA,
        "weights": {
            "score": {m: W[:, j].tolist() for j, m in enumerate(MODEL_IDS)},
            "log_cost": {m: W[:, 3 + j].tolist() for j, m in enumerate(MODEL_IDS)},
        },
        "cost_smear": {m: float(smear[j]) for j, m in enumerate(MODEL_IDS)},
        "cost_sigma": {m: float(sigma[j]) for j, m in enumerate(MODEL_IDS)},
        "tier_config": TIER_CONFIG,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
    }
    args.artifact.write_text(
        json.dumps(artifact_payload, ensure_ascii=False), encoding="utf-8"
    )
    size_mb = args.artifact.stat().st_size / 1e6
    print(f"  아티팩트 저장: {args.artifact} ({size_mb:.2f} MB)")

    print("[3/3] 런타임 대조 + train-fit 참고 수치")
    artifact = learned_router.load_artifact(args.artifact)
    sample_step = max(1, n // 64)
    worst = 0.0
    for i in range(0, n, sample_step):
        runtime_pred = learned_router.predict_episode(inputs.episodes[i], artifact)
        raw = X[i] @ W
        for j, m in enumerate(MODEL_IDS):
            score = min(1.0, max(0.0, float(raw[j])))
            log_cost = min(5.0, max(-20.0, float(raw[3 + j])))
            cost = math.exp(log_cost) * float(smear[j])
            got_score, got_cost = runtime_pred[m]
            worst = max(worst, abs(got_score - score), abs(got_cost - cost))
    if worst > 1e-9:
        raise AssertionError(f"런타임-트레이너 예측 불일치: {worst}")
    print(f"  런타임 표본 대조 통과 (최대 오차 {worst:.2e})")

    report = {"lambda": LAMBDA, "tier_config": TIER_CONFIG, "train_fit": {}}
    light_total = sum(truth[(eid, MODEL_IDS[0])][1] for eid in episode_ids)
    for tier in TIERS:
        beta = TIER_CONFIG[tier]["beta"]
        margin = TIER_CONFIG[tier]["margin"]
        mult = float(policy.tiers[tier].budget_multiplier)
        pess = np.array(
            [1.0, math.exp(beta * float(sigma[1])), math.exp(beta * float(sigma[2]))]
        )
        raw = X @ W
        scores = np.clip(raw[:, :3], 0.0, 1.0)
        costs = np.exp(np.clip(raw[:, 3:], -20.0, 5.0)) * smear[None, :] * pess[None, :]
        preds = [
            {
                MODEL_IDS[0]: (float(scores[i, 0]), float(costs[i, 0])),
                MODEL_IDS[1]: (float(scores[i, 1]), float(costs[i, 1])),
                MODEL_IDS[2]: (float(scores[i, 2]), float(costs[i, 2])),
            }
            for i in range(n)
        ]
        choice = allocate(preds, mult, margin)
        score = sum(truth[(eid, c)][0] for eid, c in zip(episode_ids, choice)) / n
        used = (
            sum(truth[(eid, c)][1] for eid, c in zip(episode_ids, choice)) / light_total
        )
        report["train_fit"][tier] = {"score": score, "used": used, "limit": mult}
        print(
            f"  [train-fit 낙관치] {tier:9s} score={score:.4f} used={used:.3f}/{mult}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  보고서 저장: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
