# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""T2 실험 v2: uplift 직접 학습 + 앙상블 — 결정량(uplift) 예측 품질로 비교.

변형:
  A ridge        (베이스라인 재현 조건)
  B gbm          (v1과 동일)
  C gbm-diff     (타깃: score_L, uplift_M, uplift_K, logcost 3종 → 재구성)
  D ens(A+B)     (head별 평균)
  E ens(A+C)

진단: OOF에서 corr(예측 uplift, 실제 uplift).
평가: Dev에서 안전계수 보정(빠른 float 채점) → 한도최대/마진90% 최종점수.
최종 승자만 공식 Decimal 채점기로 재검증한다 (별도 단계).
"""

import sys
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

import numpy as np
import lightgbm as lgb

import hash_regex
from train_hash_regex import (
    _training_matrix,
    _oof_predictions,
    _fit_ridge,
    _predict_ridge,
)
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_input,
    load_outcomes,
    load_bundled_policy,
)

policy = load_bundled_policy()
train_in = load_input(REPO / "data/materialized/train/inputs.json")
train_out = load_outcomes(REPO / "data/train/outcomes.json")
dev_in = load_input(REPO / "data/materialized/dev/inputs.json")
dev_out = load_outcomes(REPO / "data/dev/outcomes.json")

HASH_BINS = 256
X_tr, Y_tr = _training_matrix(train_in, train_out, policy, HASH_BINS)
X_dev = np.asarray(
    [hash_regex.raw_feature_vector(ep, HASH_BINS) for ep in dev_in.episodes]
)
N = X_tr.shape[0]

# Dev 실제 score·비용 (빠른 채점용)
RATES = {
    "ax31-light": (1.0, 4.0),
    "ax31": (2.127, 8.509),
    "axk1-think": (6.565, 26.260),
}
dev_truth = {}
for oc in dev_out.outcomes:
    r = RATES[oc.model_id]
    cost = (oc.input_tokens * r[0] + oc.output_tokens * r[1]) / 1e6
    dev_truth[(oc.episode_id, oc.model_id)] = (float(oc.score), cost)
dev_ids = [ep.episode_id for ep in dev_in.episodes]
true_s = {m: np.array([dev_truth[(i, m)][0] for i in dev_ids]) for m in MODEL_IDS}
true_c = {m: np.array([dev_truth[(i, m)][1] for i in dev_ids]) for m in MODEL_IDS}
light_total = float(true_c["ax31-light"].sum())

PARAMS = dict(
    objective="regression",
    num_leaves=15,
    learning_rate=0.05,
    min_data_in_leaf=25,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
    seed=7,
)
FOLDS = 5
fold_ids = np.arange(N) % FOLDS


def gbm_oof_and_dev(targets, round_grid=(50, 100, 200, 400)):
    """타깃 행렬 각 열에 대해 OOF 라운드 선택 + 전체 학습 Dev 예측."""
    T = targets.shape[1]
    oof = np.empty_like(targets)
    dev = np.empty((X_dev.shape[0], T))
    for t in range(T):
        best = None
        for rounds in round_grid:
            cur = np.empty(N)
            for f in range(FOLDS):
                va = fold_ids == f
                ds = lgb.Dataset(X_tr[~va], label=targets[~va, t])
                cur[va] = lgb.train(PARAMS, ds, num_boost_round=rounds).predict(
                    X_tr[va]
                )
            mse = float(np.mean((cur - targets[:, t]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, rounds, cur.copy())
        oof[:, t] = best[2]
        ds = lgb.Dataset(X_tr, label=targets[:, t])
        dev[:, t] = lgb.train(PARAMS, ds, num_boost_round=best[1]).predict(X_dev)
    return oof, dev


# --- A: ridge ---
ridge_oof = _oof_predictions(X_tr, Y_tr, folds=FOLDS, alpha=10.0)
mean, scale, intercept, coef = _fit_ridge(X_tr, Y_tr, 10.0)
ridge_dev = _predict_ridge(X_dev, mean, scale, intercept, coef)

# --- B: gbm plain ---
gbm_oof, gbm_dev = gbm_oof_and_dev(Y_tr)

# --- C: gbm-diff ---
diff_targets = np.column_stack(
    [
        Y_tr[:, 0],  # score_L
        Y_tr[:, 1] - Y_tr[:, 0],  # uplift_M
        Y_tr[:, 2] - Y_tr[:, 0],  # uplift_K
        Y_tr[:, 3],
        Y_tr[:, 4],
        Y_tr[:, 5],
    ]
)
diff_oof_raw, diff_dev_raw = gbm_oof_and_dev(diff_targets)


def reconstruct(raw):
    out = np.empty_like(raw)
    out[:, 0] = raw[:, 0]
    out[:, 1] = raw[:, 0] + raw[:, 1]
    out[:, 2] = raw[:, 0] + raw[:, 2]
    out[:, 3:] = raw[:, 3:]
    return out


diff_oof = reconstruct(diff_oof_raw)
diff_dev = reconstruct(diff_dev_raw)

variants = {
    "A ridge": (ridge_oof, ridge_dev),
    "B gbm": (gbm_oof, gbm_dev),
    "C gbm-diff": (diff_oof, diff_dev),
    "D ens(A+B)": ((ridge_oof + gbm_oof) / 2, (ridge_dev + gbm_dev) / 2),
    "E ens(A+C)": ((ridge_oof + diff_oof) / 2, (ridge_dev + diff_dev) / 2),
}

# --- 진단: OOF uplift 상관 ---
tu_m = Y_tr[:, 1] - Y_tr[:, 0]
tu_k = Y_tr[:, 2] - Y_tr[:, 0]
print("OOF uplift 예측 상관 (결정량 품질):")
for name, (oof, _) in variants.items():
    pu_m = oof[:, 1] - oof[:, 0]
    pu_k = oof[:, 2] - oof[:, 0]
    cm = float(np.corrcoef(pu_m, tu_m)[0, 1])
    ck = float(np.corrcoef(pu_k, tu_k)[0, 1])
    print(f"  {name:<12} corr(uplift_M)={cm:+.3f}  corr(uplift_K)={ck:+.3f}")


def rows_from(pred):
    scores, costs = [], []
    for row in pred:
        s = {m: min(1.0, max(0.0, float(row[i]))) for i, m in enumerate(MODEL_IDS)}
        c = {
            m: math.exp(min(50.0, max(-50.0, float(row[3 + i]))))
            for i, m in enumerate(MODEL_IDS)
        }
        light = c[MODEL_IDS[0]]
        c[MODEL_IDS[1]] = max(c[MODEL_IDS[1]], light * (1 + 1e-12))
        c[MODEL_IDS[2]] = max(c[MODEL_IDS[2]], c[MODEL_IDS[1]] * (1 + 1e-12))
        scores.append(s)
        costs.append(c)
    return scores, costs


def fast_eval(selected):
    s = float(np.mean([true_s[m][i] for i, m in enumerate(selected)]))
    c = float(sum(true_c[m][i] for i, m in enumerate(selected)))
    return s, c / light_total


def calibrate_fast(pred_scores, pred_costs, cap_of_limit):
    total = 0.0
    detail = {}
    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        lo = 1.0 / mult
        best = None
        for i in range(121):
            safety = lo + (1 - lo) * i / 120
            selected, _ = hash_regex.select_models(
                pred_scores, pred_costs, budget_multiplier=mult, safety_ratio=safety
            )
            score, ratio = fast_eval(selected)
            if ratio > mult * cap_of_limit + 1e-12:
                continue
            if best is None or (score, -ratio) > (best[0], -best[1]):
                best = (score, ratio, safety)
        detail[tier] = best
        total += weights[tier] * best[0]
    return total, detail


print("\nDev 최종점수 (빠른 채점, 안전계수 Dev 보정):")
print(f"  {'변형':<12}{'한도최대':>10}{'마진90%':>10}")
results = {}
for name, (_, dev) in variants.items():
    ps, pc = rows_from(dev)
    full, d_full = calibrate_fast(ps, pc, 1.0)
    m90, d_m90 = calibrate_fast(ps, pc, 0.90)
    results[name] = (full, m90, d_full, d_m90)
    print(f"  {name:<12}{full:>10.6f}{m90:>10.6f}")

best_name = max(results, key=lambda k: results[k][1])
print(f"\n마진90% 기준 최선: {best_name}")
for tier in TIERS:
    s, r, safety = results[best_name][3][tier]
    print(f"  {tier}: 점수 {s:.6f}  비용비율 {r:.4f}  안전계수 {safety:.4f}")
