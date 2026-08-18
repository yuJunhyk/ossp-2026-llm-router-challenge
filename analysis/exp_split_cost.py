# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""경로4 게이트 — 입력/출력 분리 비용 예측 vs 현행 단일 log-비용 (train 전용).

근거(2x2 분해 실측): 비용 예측 오차가 +2.5pp를 잠그고 있고, 비용의 입력
성분은 준결정적(문자수 상관 0.94+)인데 현행은 총비용 하나를 통째로 회귀해
쉬운 성분과 어려운 성분을 섞는다. 여기서는 모델별 log(입력토큰)·log(출력토큰)을
각각 회귀하고 비용식으로 정확 합성한다. smear는 성분별로 적용하고, β·σ
비관은 합성 비용의 잔차 σ(새로 측정)에 걸어 기존 캘리브레이션을 재사용한다.

게이트: 동일 프로토콜 weighted CV가 기준선(0.6553) 대비 +0.004 이상이면 채택.
v1.3 무접촉 — 산출물은 build/v14/ 전용.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np

from rematch import (
    CV_SEEDS,
    FOLDS,
    calibrate,
    episode_text,
    fit_dual_ridge,
    group_fold_ids,
    stress_masks,
    template_group_keys,
)
from os2_features import TOTAL_DIM, extract_sparse
from ossp_router.protocol import (
    MODEL_IDS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

LAM = 10.0
BASELINE_CV = 0.6553
GATE_DELTA = 0.004


def synth_cost_matrix(pred_tok, smear_in, smear_out, rates_in, rates_out):
    """(n,6)[log_in x3, log_out x3] 예측 → (n,3) 합성 기대 비용."""
    li = np.clip(pred_tok[:, :3], -5.0, 15.0)
    lo = np.clip(pred_tok[:, 3:], -5.0, 15.0)
    cost_in = np.exp(li) * smear_in[None, :] * rates_in[None, :] / 1e6
    cost_out = np.exp(lo) * smear_out[None, :] * rates_out[None, :] / 1e6
    return cost_in + cost_out


def main() -> int:
    policy = load_bundled_policy()
    inputs = load_input(REPO / "data/materialized/train/inputs.json")
    outcomes = load_outcomes(REPO / "data/train/outcomes.json")
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    rates = {m: policy.models[m] for m in MODEL_IDS}
    rates_in = np.array([float(rates[m].input_token_rate) for m in MODEL_IDS])
    rates_out = np.array([float(rates[m].output_token_rate) for m in MODEL_IDS])

    tokens = {}
    scores = {}
    for oc in outcomes.outcomes:
        tokens[(oc.episode_id, oc.model_id)] = (oc.input_tokens, oc.output_tokens)
        scores[(oc.episode_id, oc.model_id)] = float(oc.score)
    eids = [ep.episode_id for ep in inputs.episodes]

    def true_cost_of(eid, m):
        ti, to = tokens[(eid, m)]
        r = rates[m]
        return (
            float(r.fixed_cost)
            + (ti * float(r.input_token_rate) + to * float(r.output_token_rate))
            / policy.token_unit
        )

    true_score = [{m: scores[(e, m)] for m in MODEL_IDS} for e in eids]
    true_cost = [{m: true_cost_of(e, m) for m in MODEL_IDS} for e in eids]
    light_costs = [c[MODEL_IDS[0]] for c in true_cost]
    truth = {"score": true_score, "cost": true_cost, "light_costs": light_costs}

    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val
    # 타깃: 점수 3 + log입력토큰 3 + log출력토큰 3
    Y_score = np.zeros((n, 3))
    Y_tok = np.zeros((n, 6))
    log_true_cost = np.zeros((n, 3))
    for i, e in enumerate(eids):
        for j, m in enumerate(MODEL_IDS):
            Y_score[i, j] = scores[(e, m)]
            ti, to = tokens[(e, m)]
            Y_tok[i, j] = math.log(max(ti, 1))
            Y_tok[i, 3 + j] = math.log(max(to, 1))
            log_true_cost[i, j] = math.log(max(true_cost[i][m], 1e-9))
    Y_all = np.hstack([Y_score, Y_tok])  # (n, 9)

    keys = template_group_keys(texts)
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    masks = stress_masks(texts)

    bundles, oof_by_seed = [], []
    sigma_report = []
    for fold_of in fold_sets:
        raw_full = np.zeros((n, 6))
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            W = fit_dual_ridge(X[tr], Y_all[tr], LAM)
            pred_tr = X[tr] @ W
            pred_va = X[va] @ W
            # 성분별 smear (fold-train 잔차)
            smear_in = np.exp(Y_tok[tr][:, :3] - pred_tr[:, 3:6]).mean(axis=0)
            smear_out = np.exp(Y_tok[tr][:, 3:] - pred_tr[:, 6:9]).mean(axis=0)
            # 합성 비용 + 합성 잔차 σ (fold-train)
            synth_tr = synth_cost_matrix(
                pred_tr[:, 3:9], smear_in, smear_out, rates_in, rates_out
            )
            resid = log_true_cost[tr] - np.log(np.maximum(synth_tr, 1e-12))
            sigma = resid.std(axis=0)
            smear = np.exp(resid).mean(axis=0)  # 잔여 편향 보정 (합성 후)
            sigma_acc.append(sigma)
            smear_acc.append(smear)
            sigma_report.append(sigma)

            synth_va = synth_cost_matrix(
                pred_va[:, 3:9], smear_in, smear_out, rates_in, rates_out
            )
            pred6 = np.hstack([pred_va[:, :3], np.log(np.maximum(synth_va, 1e-12))])
            raw_full[va] = pred6
            bundles.append(
                {
                    "idx": np.where(va)[0],
                    "pred": pred6,
                    "smear": smear,
                    "sigma": sigma,
                    "light_base": sum(light_costs[i] for i in np.where(va)[0]),
                }
            )
        oof_by_seed.append(
            {
                "pred": raw_full,
                "smear": np.mean(smear_acc, axis=0),
                "sigma": np.mean(sigma_acc, axis=0),
            }
        )

    sig = np.mean(sigma_report, axis=0)
    print(
        f"합성 비용 log-잔차 σ (light/ax31/think): {sig[0]:.3f} / {sig[1]:.3f} / {sig[2]:.3f}"
        f"  (현행 단일 head: 0.618 / 0.497 / 0.680)"
    )

    res = calibrate("linear+split-cost", bundles, oof_by_seed, masks, truth, policy)
    delta = res["weighted"] - BASELINE_CV
    print(f"weighted_cv={res['weighted']:.4f}  (기준선 {BASELINE_CV}, Δ={delta:+.4f})")
    for tier, cfg in res["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )
    verdict = delta >= GATE_DELTA
    print(
        f"경로4 게이트: {'통과 → 채택' if verdict else '미달 → 기각 (현행 비용 head 유지)'}"
    )

    out = REPO / "build/v14/split-cost.json"
    out.write_text(
        json.dumps(
            {
                "weighted": res["weighted"],
                "delta_vs_baseline": delta,
                "pass": verdict,
                "sigma_synth": sig.tolist(),
                "tiers": res["tiers"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
