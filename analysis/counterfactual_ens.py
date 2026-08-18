# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""반사실 실측 — "v1.2(ens 계열)를 dev 없이 만들었다면 dev 점수는?"

v1.2의 dev 0.6920은 dev를 선택에 재사용한 낙관 편향 수치다. 여기서는
재대결(analysis/rematch.py)이 dev-clean 프로토콜로 재선택해 둔 ens 구성
두 가지를 전체 train으로 최종 적합해 dev에 1회씩 적용, "오염의 크기"를
실측한다.

- ens(w=1.0): dev 없이 진행했다면 train CV가 실제로 골랐을 ens 구성
- ens(w=0.5): v1.2의 실제 앙상블 가중에서 dev 튜닝만 제거한 구성

주의: 이 결과는 회고 기록 전용이다. 어떤 향후 선택의 근거로도 쓰지 않는다
(쓰는 순간 dev 오염이 재발한다). 동결된 v1.3 런타임·아티팩트는 건드리지
않으며, 산출물은 build/counterfactual/ 아래에만 쓴다.

GBM 라운드와 tier별 (β, margin)은 build/rematch-report.json의 값을 그대로
고정 사용한다 — 여기서 재선택하면 반사실이 아니라 새 튜닝이 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np
import lightgbm as lgb

from train_hash_regex import _fit_ridge, _predict_ridge, _training_matrix
from train_ens import GBM_PARAMS, HASH_BINS, RIDGE_ALPHA, to_diff, reconstruct
from rematch import cost_stats, rows_with_pessimism
from os2_policy import allocate
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    submission_to_dict,
)

GBM_SEEDS = (7, 8, 9)
# 재대결(그룹 OOF)에서 확정된 값들 — 고정, 재선택 금지
ROUNDS = (200, 100, 200, 1200, 800, 400)
VARIANTS = {
    "1.0": {
        "fast": (1.5, 0.92),
        "balanced": (1.5, 0.84),
        "premium": (1.5, 0.80),
    },
    "0.5": {
        "fast": (1.5, 0.94),
        "balanced": (1.0, 0.94),
        "premium": (1.5, 0.84),
    },
}


def realized(episode_ids, choice, truth):
    n = len(episode_ids)
    score = sum(truth[(eid, c)][0] for eid, c in zip(episode_ids, choice)) / n
    spent = sum(truth[(eid, c)][1] for eid, c in zip(episode_ids, choice))
    light = sum(truth[(eid, MODEL_IDS[0])][1] for eid in episode_ids)
    return score, spent / light


def truth_map(outcomes, policy):
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
    return truth


def main() -> int:
    parser = argparse.ArgumentParser(description="dev-clean ens 반사실 실측")
    parser.add_argument("--weight", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="제출 출력 디렉토리 (기본 build/counterfactual/w<weight>)",
    )
    args = parser.parse_args()
    w = float(args.weight)
    tier_config = VARIANTS[args.weight]
    out_dir = (
        args.out
        or REPO / "build" / "counterfactual" / f"w{args.weight.replace('.', '')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = load_bundled_policy()
    train_inputs = load_input(REPO / "data/materialized/train/inputs.json")
    train_outcomes = load_outcomes(REPO / "data/train/outcomes.json")
    dev_inputs = load_input(REPO / "data/materialized/dev/inputs.json")
    dev_outcomes = load_outcomes(REPO / "data/dev/outcomes.json")

    print(f"[1/3] 특징 행렬 (w={w})")
    X_tr, Y_tr = _training_matrix(train_inputs, train_outcomes, policy, HASH_BINS)
    X_dev, _ = _training_matrix(dev_inputs, dev_outcomes, policy, HASH_BINS)
    diff = to_diff(Y_tr)

    print("[2/3] 전체 train 적합 (ridge + GBM, 재대결 확정 라운드)")
    m, s, b, c = _fit_ridge(X_tr, Y_tr, RIDGE_ALPHA)
    ridge_tr = _predict_ridge(X_tr, m, s, b, c)
    ridge_dev = _predict_ridge(X_dev, m, s, b, c)
    gbm_tr = np.zeros_like(diff)
    gbm_dev = np.zeros((X_dev.shape[0], diff.shape[1]))
    for t in range(diff.shape[1]):
        for seed in GBM_SEEDS:
            ds = lgb.Dataset(X_tr, label=diff[:, t])
            booster = lgb.train(
                dict(GBM_PARAMS, seed=seed), ds, num_boost_round=ROUNDS[t]
            )
            gbm_tr[:, t] += booster.predict(X_tr) / len(GBM_SEEDS)
            gbm_dev[:, t] += booster.predict(X_dev) / len(GBM_SEEDS)

    pred_tr = (1 - w) * ridge_tr + w * reconstruct(gbm_tr)
    pred_dev = (1 - w) * ridge_dev + w * reconstruct(gbm_dev)
    smear, sigma = cost_stats(pred_tr[:, 3:], Y_tr[:, 3:])

    print("[3/3] dev 배분·제출 생성 + 실현 수치")
    train_truth = truth_map(train_outcomes, policy)
    dev_truth = truth_map(dev_outcomes, policy)
    train_ids = [ep.episode_id for ep in train_inputs.episodes]
    dev_ids = [ep.episode_id for ep in dev_inputs.episodes]

    summary = {"weight": w, "rounds": list(ROUNDS), "tiers": {}}
    for tier in TIERS:
        beta, margin = tier_config[tier]
        mult = float(policy.tiers[tier].budget_multiplier)
        # 참고용: train 적합 재예측(낙관치)
        rows_tr = rows_with_pessimism(pred_tr, smear, sigma, beta)
        s_tr, u_tr = realized(train_ids, allocate(rows_tr, mult, margin), train_truth)
        # 반사실 본판: dev
        rows_dev = rows_with_pessimism(pred_dev, smear, sigma, beta)
        choice = allocate(rows_dev, mult, margin)
        s_dev, u_dev = realized(dev_ids, choice, dev_truth)
        summary["tiers"][tier] = {
            "beta": beta,
            "margin": margin,
            "train_fit_score": s_tr,
            "train_fit_used": u_tr,
            "dev_score": s_dev,
            "dev_used": u_dev,
            "budget_multiplier": mult,
            "over_budget": u_dev > mult,
        }
        print(
            f"  {tier:9s} dev score={s_dev:.4f} used={u_dev:.3f}/{mult}"
            f"{'  ** 예산 초과 **' if u_dev > mult else ''}"
            f"  (train-fit {s_tr:.4f}, {u_tr:.3f})"
        )
        submission = Submission(
            schema_version=dev_inputs.schema_version,
            challenge_id=dev_inputs.challenge_id,
            policy_id=policy.policy_id,
            split=dev_inputs.split,
            tier=tier,
            decisions=tuple(
                Decision(eid, model_id) for eid, model_id in zip(dev_ids, choice)
            ),
        )
        (out_dir / f"{tier}.json").write_text(
            json.dumps(submission_to_dict(submission), ensure_ascii=False),
            encoding="utf-8",
        )

    weights = {t: float(policy.tiers[t].weight) for t in TIERS}
    final = sum(weights[t] * summary["tiers"][t]["dev_score"] for t in TIERS)
    zeroed = sum(
        weights[t]
        * (
            0.0
            if summary["tiers"][t]["over_budget"]
            else summary["tiers"][t]["dev_score"]
        )
        for t in TIERS
    )
    summary["dev_final_ignoring_budget"] = final
    summary["dev_final_with_zero_rule"] = zeroed
    print(f"  final (예산 0점 규칙 적용): {zeroed:.4f}  /  미적용 합산: {final:.4f}")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"제출·요약 저장: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
