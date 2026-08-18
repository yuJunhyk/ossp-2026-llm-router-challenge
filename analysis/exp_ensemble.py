# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""경로3 게이트 — 이종 특징 점수 앙상블: linear(8,232) × GBM-diff(270) (train 전용).

점수 head만 앙상블한다: score = (1-w)·linear + w·GBM-diff 재구성, w ∈ {0.25, 0.5}.
GBM은 재대결 확정 라운드(score 타깃: 200/100/200)와 seed(7,8,9)를 고정 사용.
비용 head는 S3 승자(--use-split-cost 시 분리 합성, 아니면 현행 단일 log-비용).

게이트: S3 승자 CV 대비 ΔCV ≥ +0.004이면 채택. v1.3 무접촉.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np
import lightgbm as lgb

from train_hash_regex import _training_matrix
from train_ens import GBM_PARAMS, HASH_BINS, to_diff
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
from exp_split_cost import synth_cost_matrix
from os2_features import TOTAL_DIM, extract_sparse
from ossp_router.protocol import (
    MODEL_IDS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

LAM = 10.0
GBM_SEEDS = (7, 8, 9)
SCORE_ROUNDS = (200, 100, 200)  # 재대결 확정 (score_light, uplift_M, uplift_K)
WEIGHTS_GRID = (0.25, 0.5)
GATE_DELTA = 0.004


def recon_scores(diff_pred):
    """diff (n,3) → 모델별 점수 (n,3)."""
    out = np.empty_like(diff_pred)
    out[:, 0] = diff_pred[:, 0]
    out[:, 1] = diff_pred[:, 0] + diff_pred[:, 1]
    out[:, 2] = diff_pred[:, 0] + diff_pred[:, 2]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="이종 특징 점수 앙상블 게이트")
    parser.add_argument("--use-split-cost", action="store_true")
    parser.add_argument(
        "--baseline-cv", type=float, required=True, help="S3 승자 weighted CV"
    )
    args = parser.parse_args()

    policy = load_bundled_policy()
    inputs = load_input(REPO / "data/materialized/train/inputs.json")
    outcomes = load_outcomes(REPO / "data/train/outcomes.json")
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    rates = {m: policy.models[m] for m in MODEL_IDS}
    rates_in = np.array([float(rates[m].input_token_rate) for m in MODEL_IDS])
    rates_out = np.array([float(rates[m].output_token_rate) for m in MODEL_IDS])
    tokens, scores_map = {}, {}
    for oc in outcomes.outcomes:
        tokens[(oc.episode_id, oc.model_id)] = (oc.input_tokens, oc.output_tokens)
        scores_map[(oc.episode_id, oc.model_id)] = float(oc.score)
    eids = [ep.episode_id for ep in inputs.episodes]

    def true_cost_of(eid, m):
        ti, to = tokens[(eid, m)]
        r = rates[m]
        return (
            float(r.fixed_cost)
            + (ti * float(r.input_token_rate) + to * float(r.output_token_rate))
            / policy.token_unit
        )

    true_score = [{m: scores_map[(e, m)] for m in MODEL_IDS} for e in eids]
    true_cost = [{m: true_cost_of(e, m) for m in MODEL_IDS} for e in eids]
    light_costs = [c[MODEL_IDS[0]] for c in true_cost]
    truth = {"score": true_score, "cost": true_cost, "light_costs": light_costs}

    # ── 선형 특징·타깃 (9-head: 점수3 + log입력3 + log출력3)
    X_lin = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X_lin[i, idx] = val
    Y_score = np.zeros((n, 3))
    Y_tok = np.zeros((n, 6))
    log_true_cost = np.zeros((n, 3))
    log_total = np.zeros((n, 3))
    for i, e in enumerate(eids):
        for j, m in enumerate(MODEL_IDS):
            Y_score[i, j] = scores_map[(e, m)]
            ti, to = tokens[(e, m)]
            Y_tok[i, j] = math.log(max(ti, 1))
            Y_tok[i, 3 + j] = math.log(max(to, 1))
            log_true_cost[i, j] = math.log(max(true_cost[i][m], 1e-9))
            log_total[i, j] = log_true_cost[i, j]
    Y_all = np.hstack(
        [Y_score, Y_tok, log_total]
    )  # (n, 12) — 마지막 3은 단일 비용 head용

    # ── GBM 특징 (270) + diff 점수 타깃
    X_gbm, Y6 = _training_matrix(inputs, outcomes, policy, HASH_BINS)
    diff = to_diff(Y6)[:, :3]

    keys = template_group_keys(texts)
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    masks = stress_masks(texts)

    # fold별 원자재: 선형 예측 + GBM 점수 diff 예측
    fold_data = []
    for fold_of in fold_sets:
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            W = fit_dual_ridge(X_lin[tr], Y_all[tr], LAM)
            pred_tr = X_lin[tr] @ W
            pred_va = X_lin[va] @ W
            gbm_va = np.zeros((int(va.sum()), 3))
            for t in range(3):
                for seed in GBM_SEEDS:
                    ds = lgb.Dataset(X_gbm[tr], label=diff[tr][:, t])
                    booster = lgb.train(
                        dict(GBM_PARAMS, seed=seed), ds, num_boost_round=SCORE_ROUNDS[t]
                    )
                    gbm_va[:, t] += booster.predict(X_gbm[va]) / len(GBM_SEEDS)
            fold_data.append(
                {
                    "va": np.where(va)[0],
                    "tr": np.where(tr)[0],
                    "lin_tr": pred_tr,
                    "lin_va": pred_va,
                    "gbm_va_scores": recon_scores(gbm_va),
                }
            )
            print(f"  fold {len(fold_data)}/15 완료", flush=True)

    results = {}
    for w in WEIGHTS_GRID:
        bundles, oof_by_seed = [], []
        for si in range(len(CV_SEEDS)):
            raw_full = np.zeros((n, 6))
            smear_acc, sigma_acc = [], []
            for k in range(FOLDS):
                f = fold_data[si * FOLDS + k]
                tr, va = f["tr"], f["va"]
                score_va = (1 - w) * f["lin_va"][:, :3] + w * f["gbm_va_scores"]
                if args.use_split_cost:
                    smear_in = np.exp(Y_tok[tr][:, :3] - f["lin_tr"][:, 3:6]).mean(
                        axis=0
                    )
                    smear_out = np.exp(Y_tok[tr][:, 3:] - f["lin_tr"][:, 6:9]).mean(
                        axis=0
                    )
                    synth_tr = synth_cost_matrix(
                        f["lin_tr"][:, 3:9], smear_in, smear_out, rates_in, rates_out
                    )
                    resid = log_true_cost[tr] - np.log(np.maximum(synth_tr, 1e-12))
                    synth_va = synth_cost_matrix(
                        f["lin_va"][:, 3:9], smear_in, smear_out, rates_in, rates_out
                    )
                    cost_log_va = np.log(np.maximum(synth_va, 1e-12))
                else:
                    resid = log_true_cost[tr] - f["lin_tr"][:, 9:12]
                    cost_log_va = f["lin_va"][:, 9:12]
                sigma = resid.std(axis=0)
                smear = np.exp(resid).mean(axis=0)
                pred6 = np.hstack([score_va, cost_log_va])
                raw_full[va] = pred6
                smear_acc.append(smear)
                sigma_acc.append(sigma)
                bundles.append(
                    {
                        "idx": va,
                        "pred": pred6,
                        "smear": smear,
                        "sigma": sigma,
                        "light_base": sum(light_costs[i] for i in va),
                    }
                )
            oof_by_seed.append(
                {
                    "pred": raw_full,
                    "smear": np.mean(smear_acc, axis=0),
                    "sigma": np.mean(sigma_acc, axis=0),
                }
            )
        res = calibrate(f"ensemble(w={w})", bundles, oof_by_seed, masks, truth, policy)
        results[w] = res
        print(f"w={w}: weighted_cv={res['weighted']:.4f}", flush=True)

    best_w = max(results, key=lambda w: results[w]["weighted"])
    best = results[best_w]
    delta = best["weighted"] - args.baseline_cv
    verdict = delta >= GATE_DELTA
    print(
        f"\n최고 w={best_w}: {best['weighted']:.4f} (S3 승자 {args.baseline_cv}, Δ={delta:+.4f})"
    )
    print(f"경로3 게이트: {'통과 → 채택' if verdict else '미달 → 기각 (앙상블 없음)'}")

    out = REPO / "build/v14/ensemble.json"
    out.write_text(
        json.dumps(
            {
                "use_split_cost": args.use_split_cost,
                "baseline_cv": args.baseline_cv,
                "results": {str(w): r["weighted"] for w, r in results.items()},
                "best_w": best_w,
                "delta": delta,
                "pass": verdict,
                "best_tiers": best["tiers"],
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
