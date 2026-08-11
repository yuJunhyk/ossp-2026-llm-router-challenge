# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""T2 실험 v4: gbm-diff 하이퍼파라미터 탐색 (선택은 train OOF로만).

- 탐색: num_leaves × min_data_in_leaf × learning_rate
- 선택 기준: OOF 목적함수 = score부 MSE + 0.05 × logcost부 MSE (공식 트레이너와 동일 가중)
- 보고: OOF 상위 3개 구성의 Dev 마진90% 점수 (ridge와 w=0.5 앙상블)
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
from train_hash_regex import _training_matrix, _fit_ridge, _predict_ridge
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
FOLDS = 5
fold_ids = np.arange(N) % FOLDS

diff_targets = Y_tr.copy()
diff_targets[:, 1] = Y_tr[:, 1] - Y_tr[:, 0]
diff_targets[:, 2] = Y_tr[:, 2] - Y_tr[:, 0]

RATES = {
    "ax31-light": (1.0, 4.0),
    "ax31": (2.127, 8.509),
    "axk1-think": (6.565, 26.260),
}
dev_truth = {}
for oc in dev_out.outcomes:
    r = RATES[oc.model_id]
    dev_truth[(oc.episode_id, oc.model_id)] = (
        float(oc.score),
        (oc.input_tokens * r[0] + oc.output_tokens * r[1]) / 1e6,
    )
dev_ids = [ep.episode_id for ep in dev_in.episodes]
true_s = {m: np.array([dev_truth[(i, m)][0] for i in dev_ids]) for m in MODEL_IDS}
true_c = {m: np.array([dev_truth[(i, m)][1] for i in dev_ids]) for m in MODEL_IDS}
light_total = float(true_c["ax31-light"].sum())


def run_config(leaves, min_data, lr):
    rounds_grid = (100, 200, 400) if lr >= 0.05 else (200, 400, 800)
    params = dict(
        objective="regression",
        num_leaves=leaves,
        learning_rate=lr,
        min_data_in_leaf=min_data,
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
    oof = np.empty_like(diff_targets)
    best_rounds = []
    for t in range(6):
        best = None
        for rounds in rounds_grid:
            cur = np.empty(N)
            for f in range(FOLDS):
                va = fold_ids == f
                ds = lgb.Dataset(X_tr[~va], label=diff_targets[~va, t])
                cur[va] = lgb.train(params, ds, num_boost_round=rounds).predict(
                    X_tr[va]
                )
            mse = float(np.mean((cur - diff_targets[:, t]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, rounds, cur.copy())
        best_rounds.append(best[1])
        oof[:, t] = best[2]
    score_mse = float(np.mean((oof[:, :3] - diff_targets[:, :3]) ** 2))
    cost_mse = float(np.mean((oof[:, 3:] - diff_targets[:, 3:]) ** 2))
    objective = score_mse + 0.05 * cost_mse
    return objective, best_rounds, params


configs = []
for leaves in (7, 15, 31):
    for min_data in (15, 25, 50):
        for lr in (0.03, 0.05):
            obj, rounds, params = run_config(leaves, min_data, lr)
            configs.append((obj, leaves, min_data, lr, rounds, params))
            print(
                f"leaves={leaves:<3} min_data={min_data:<3} lr={lr}: OOF목적 {obj:.5f}"
            )

configs.sort(key=lambda c: c[0])
print("\nOOF 상위 3개 구성의 Dev 마진90% (ridge와 0.5 앙상블):")

ridge_dev = None
mean, scale, intercept, coef = _fit_ridge(X_tr, Y_tr, 10.0)
ridge_dev = _predict_ridge(X_dev, mean, scale, intercept, coef)


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


def margin90_final(pred):
    ps, pc = rows_from(pred)
    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
    total = 0.0
    detail = {}
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        lo = 1.0 / mult
        best = None
        for i in range(121):
            safety = lo + (1 - lo) * i / 120
            selected, _ = hash_regex.select_models(
                ps, pc, budget_multiplier=mult, safety_ratio=safety
            )
            s = float(np.mean([true_s[m][j] for j, m in enumerate(selected)]))
            ratio = (
                float(sum(true_c[m][j] for j, m in enumerate(selected))) / light_total
            )
            if ratio > mult * 0.90 + 1e-12:
                continue
            if best is None or (s, -ratio) > (best[0], -best[1]):
                best = (s, ratio, safety)
        detail[tier] = best
        total += weights[tier] * best[0]
    return total, detail


for obj, leaves, min_data, lr, rounds, params in configs[:3]:
    dev_pred = np.empty((X_dev.shape[0], 6))
    for t in range(6):
        ds = lgb.Dataset(X_tr, label=diff_targets[:, t])
        dev_pred[:, t] = lgb.train(params, ds, num_boost_round=rounds[t]).predict(X_dev)
    dev_pred[:, 1] += dev_pred[:, 0]
    dev_pred[:, 2] += dev_pred[:, 0]
    ens = 0.5 * ridge_dev + 0.5 * dev_pred
    total, detail = margin90_final(ens)
    print(f"  leaves={leaves} min_data={min_data} lr={lr} rounds={rounds}: {total:.6f}")
    for tier in TIERS:
        s, r, safety = detail[tier]
        print(f"    {tier}: {s:.6f} | 비용비율 {r:.4f}")
