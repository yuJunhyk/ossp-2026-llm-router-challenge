# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""선형(dual ridge) 라우터 최종 트레이너 — learned-router.v1.json 생성.

선택 근거 (train 전용 재대결, Dev 미사용 — 결정 기록은 docs/decisions.md):
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

from os2_features import KFEAT_DIM, KFEAT_SLOT, TOTAL_DIM, extract_sparse, k_guard
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


def episode_text(episode) -> str:
    if episode.prompt is not None:
        return episode.prompt
    parts = []
    for message in episode.messages or ():
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def fit_dual_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """w = X^T (XX^T + lam I)^{-1} Y (독립 재구현 세션 train_router.py 이식)."""
    n = X.shape[0]
    K = X @ X.T
    K[np.diag_indices(n)] += lam
    alpha = np.linalg.solve(K, Y)
    return X.T @ alpha


def cost_stats(
    pred_log: np.ndarray, true_log: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """log-비용 잔차의 smear(기대값 보정)와 σ (모델별)."""
    resid = true_log - pred_log
    return np.exp(resid).mean(axis=0), resid.std(axis=0)


LAMBDA = 10.0
# v1.6 uplift 축소:
# γ=0 — 3.1 점수 헤드를 Light 헤드 + 전체 train 평균 uplift 상수로 교체.
# 예측된 L→M 격차(참값 상관 0.033)가 그리디 배분에 역선택을 일으키는 것을 차단.
GAMMA = 0.0
BIAS_SLOT = 33  # os2_features numeric bias (항상 1.0)
# R1.4 (2026-08-25): 전 tier β=1.0 통일 — β0.5 얇은 쐐기의
# 독립 fold 부도 5회 실증에 따른 원칙. margin은 β1.0 쐐기가 비용을 부풀린 가격표 기준이라
# 1을 넘을 수 있다(balanced 1.08). premium은 margin(n) 분기: 배치 n>=800이면 폭탄 동거
# 위험이 희석되어(800/880/1,760 부트스트랩 전 표본 무초과) 깊은 margin 1.08을 쓴다.
TIER_CONFIG = {
    "fast": {"beta": 1.0, "margin": 0.94},
    "balanced": {"beta": 1.0, "margin": 1.08},
    "premium": {
        "beta": 1.0,
        "margin": 0.96,
        "margin_deep": 1.08,
        "deep_min_episodes": 800,
    },
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

    print(
        f"[2/3] 전체 train 적합 (λ={LAMBDA}) + v1.6 uplift 축소(γ={GAMMA}) + 아티팩트 생성"
    )
    # v1.7: KFEAT 열(40..66)을 train 평균·표준편차로 z-점수화해 적합한 뒤 스케일을 가중치에 접는다
    # (런타임은 원값 특징만 계산 — 상수 불필요). W_raw = W_z/σ, bias −= Σ μ/σ·W_z.
    kf = slice(KFEAT_SLOT, KFEAT_SLOT + KFEAT_DIM)
    mu = X[:, kf].mean(axis=0)
    sd = X[:, kf].std(axis=0) + 1e-9
    Xz = X.copy()
    Xz[:, kf] = (X[:, kf] - mu) / sd
    W = fit_dual_ridge(Xz, Y, LAMBDA)
    W[BIAS_SLOT, :] -= (mu / sd) @ W[kf, :]
    W[kf, :] = W[kf, :] / sd[:, None]
    print(f"  KFEAT z-점수 접기 완료 (μ/σ {KFEAT_DIM}개, 가중치 내장)")
    # v1.6: ŝ_M' = ŝ_L + γ(ŝ_M − ŝ_L) + (1−γ)·ḡ — 가중치에 접기 (런타임 무변경)
    gbar = float((Y[:, 1] - Y[:, 0]).mean())
    W[:, 1] = GAMMA * W[:, 1] + (1.0 - GAMMA) * W[:, 0]
    W[BIAS_SLOT, 1] += (1.0 - GAMMA) * gbar
    print(f"  전체 train 평균 uplift ḡ = {gbar:+.4f} (bias 슬롯 {BIAS_SLOT}에 반영)")
    smear, sigma = cost_stats((X @ W)[:, 3:], Y[:, 3:])

    artifact_payload = {
        "artifact_type": learned_router.ARTIFACT_TYPE,
        "schema_version": 1,
        "trained_on": (
            "public train split only (1,760 episodes); predictor and calibration selected "
            "by template-group 5-fold x 3-seed cross-validation, dev untouched; "
            "ax31 score head replaced by light head plus constant uplift; "
            "27 dense descriptive features (numeric slots 40..66, z-score folded into weights) "
            "+ runtime guard for primality/factorization prompts and "
            "large-integer polynomial equations"
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
        for i, t in enumerate(texts):
            if k_guard(t):
                preds[i][MODEL_IDS[2]] = (
                    preds[i][MODEL_IDS[1]][0],
                    preds[i][MODEL_IDS[2]][1],
                )
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
