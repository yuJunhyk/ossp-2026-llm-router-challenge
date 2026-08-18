# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""경로1 게이트 — 학습곡선: 데이터를 늘리면 점수가 오르는 구간인가 (train 전용).

각 그룹 fold의 fold-train을 그룹 단위로 {50%, 75%, 100%} 부분표본해
linear(λ=10)를 적합하고, 동일 캘리브레이션 프로토콜로 weighted CV를 잰다.
유효 학습량 704 → 1,056 → 1,408에서 곡선이 단조 상승하고 마지막 구간이
+0.003 이상이면, train+dev(2,640) 재적합이 이득일 것으로 판정한다.
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
    cost_stats,
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

FRACTIONS = (0.5, 0.75, 1.0)
LAM = 10.0


def subsample_groups(indices, keys, frac, rng):
    """fold-train 안에서 그룹 단위로 frac 비율만큼 표본."""
    if frac >= 1.0:
        return indices
    members = {}
    for i in indices:
        members.setdefault(keys[i], []).append(i)
    order = sorted(members)
    rng.shuffle(order)
    target = int(round(len(indices) * frac))
    chosen = []
    for key in order:
        if len(chosen) >= target:
            break
        chosen.extend(members[key])
    return np.array(sorted(chosen))


def main() -> int:
    policy = load_bundled_policy()
    inputs = load_input(REPO / "data/materialized/train/inputs.json")
    outcomes = load_outcomes(REPO / "data/train/outcomes.json")
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    rates = {m: policy.models[m] for m in MODEL_IDS}
    truth_pairs = {}
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
        truth_pairs[(oc.episode_id, oc.model_id)] = (float(oc.score), cost)
    eids = [ep.episode_id for ep in inputs.episodes]
    true_score = [{m: truth_pairs[(e, m)][0] for m in MODEL_IDS} for e in eids]
    true_cost = [{m: truth_pairs[(e, m)][1] for m in MODEL_IDS} for e in eids]
    light_costs = [c[MODEL_IDS[0]] for c in true_cost]
    truth = {"score": true_score, "cost": true_cost, "light_costs": light_costs}

    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val
    Y = np.zeros((n, 6))
    for i, e in enumerate(eids):
        for j, m in enumerate(MODEL_IDS):
            Y[i, j] = truth_pairs[(e, m)][0]
            Y[i, 3 + j] = math.log(max(truth_pairs[(e, m)][1], 1e-9))

    keys = template_group_keys(texts)
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    masks = stress_masks(texts)

    curve = []
    for frac in FRACTIONS:
        bundles, oof_by_seed = [], []
        eff_sizes = []
        for si, fold_of in enumerate(fold_sets):
            raw_full = np.zeros_like(Y)
            smear_acc, sigma_acc = [], []
            for k in range(FOLDS):
                va = fold_of == k
                tr_idx = np.where(~va)[0]
                rng = np.random.default_rng(9000 + si * 100 + k * 10 + int(frac * 4))
                sub = subsample_groups(tr_idx, keys, frac, rng)
                eff_sizes.append(len(sub))
                W = fit_dual_ridge(X[sub], Y[sub], LAM)
                pred_tr = X[sub] @ W
                smear, sigma = cost_stats(pred_tr[:, 3:], Y[sub][:, 3:])
                pred_va = X[va] @ W
                raw_full[va] = pred_va
                smear_acc.append(smear)
                sigma_acc.append(sigma)
                bundles.append(
                    {
                        "idx": np.where(va)[0],
                        "pred": pred_va,
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
        res = calibrate(
            f"linear(frac={frac})", bundles, oof_by_seed, masks, truth, policy
        )
        mean_n = int(np.mean(eff_sizes))
        curve.append(
            {
                "frac": frac,
                "mean_train_n": mean_n,
                "weighted": res["weighted"],
                "tiers": res["tiers"],
            }
        )
        print(
            f"frac={frac} (평균 학습 {mean_n}문항): weighted_cv={res['weighted']:.4f}",
            flush=True,
        )

    d_last = curve[-1]["weighted"] - curve[-2]["weighted"]
    monotone = all(
        curve[i]["weighted"] <= curve[i + 1]["weighted"] for i in range(len(curve) - 1)
    )
    verdict = monotone and d_last >= 0.003
    print(f"\n단조 상승={monotone}, 마지막 구간 Δ={d_last:+.4f}")
    print(
        f"경로1 게이트: {'통과 → train+dev 재적합 채택 예약' if verdict else '미달 → 재적합 폐기 (train 1,760 유지)'}"
    )

    out = REPO / "build/v14/learning-curve.json"
    out.write_text(
        json.dumps(
            {
                "curve": curve,
                "monotone": monotone,
                "delta_last": d_last,
                "pass": verdict,
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
