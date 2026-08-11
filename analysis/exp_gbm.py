# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""T2 실험: GBM(LightGBM) vs ridge — 같은 특징, Dev 점수·마진 정책 비교.

- 특징: 공식 hash-regex와 동일 (14 dense + 256 hash bins)
- 타깃: 모델별 score 3개 + log-cost 3개 (동일)
- 선택: 공식 select_models(라그랑주) 재사용
- 비교: (a) 한도-최대 보정 (공식 방법론과 동일 조건), (b) 한도 90% 마진 보정
"""

import sys
import math
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

import numpy as np
import lightgbm as lgb

import hash_regex
from train_hash_regex import _training_matrix, _score_one_tier
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
    [hash_regex.raw_feature_vector(ep, HASH_BINS) for ep in dev_in.episodes],
    dtype=np.float64,
)
N, T = X_tr.shape[0], Y_tr.shape[1]

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
ROUND_GRID = (100, 200, 400)
FOLDS = 5

# --- 타깃별 부스팅 라운드 선택 (5-fold OOF MSE) ---
fold_ids = np.arange(N) % FOLDS
best_rounds = []
oof_final = np.empty_like(Y_tr)
for t in range(T):
    best = None
    for rounds in ROUND_GRID:
        oof = np.empty(N)
        for f in range(FOLDS):
            va = fold_ids == f
            ds = lgb.Dataset(X_tr[~va], label=Y_tr[~va, t])
            booster = lgb.train(PARAMS, ds, num_boost_round=rounds)
            oof[va] = booster.predict(X_tr[va])
        mse = float(np.mean((oof - Y_tr[:, t]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, rounds, oof.copy())
    best_rounds.append(best[1])
    oof_final[:, t] = best[2]
    kind = "score" if t < 3 else "logcost"
    ridge_mse = None
    print(
        f"  target {t} ({kind} {MODEL_IDS[t % 3]}): rounds={best[1]} OOF-MSE={best[0]:.4f}"
    )

# 참고: ridge OOF와 비교
from train_hash_regex import _oof_predictions

ridge_oof = _oof_predictions(X_tr, Y_tr, folds=FOLDS, alpha=10.0)
print("\nOOF MSE 비교 (GBM vs ridge alpha=10):")
for t in range(T):
    g = float(np.mean((oof_final[:, t] - Y_tr[:, t]) ** 2))
    r = float(np.mean((ridge_oof[:, t] - Y_tr[:, t]) ** 2))
    kind = "score" if t < 3 else "logcost"
    print(
        f"  {kind:<8}{MODEL_IDS[t % 3]:<12} GBM {g:.4f}  ridge {r:.4f}  ({'GBM 우세' if g < r else 'ridge 우세'})"
    )

# --- 전체 train으로 최종 학습 → Dev 예측 ---
dev_pred = np.empty((X_dev.shape[0], T))
for t in range(T):
    ds = lgb.Dataset(X_tr, label=Y_tr[:, t])
    booster = lgb.train(PARAMS, ds, num_boost_round=best_rounds[t])
    dev_pred[:, t] = booster.predict(X_dev)


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


def calibrate(pred_scores, pred_costs, max_ratio_of_limit):
    """Dev에서 tier별 안전계수 선택. actual_ratio ≤ multiplier×max_ratio_of_limit 제약."""
    results = {}
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        lo = 1.0 / mult
        candidates = [lo + (1 - lo) * i / 120 for i in range(121)]
        best = None
        for safety in candidates:
            selected, _ = hash_regex.select_models(
                pred_scores, pred_costs, budget_multiplier=mult, safety_ratio=safety
            )
            rep = _score_one_tier(dev_in, dev_out, policy, tier, selected)
            ratio = float(Decimal(rep["budget_ratio"]))
            if ratio > mult * max_ratio_of_limit:
                continue
            rank = (Decimal(rep["tier_score"]), -Decimal(rep["budget_ratio"]))
            if best is None or rank > best[0]:
                best = (rank, safety, rep)
        results[tier] = best
    weights = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
    final = sum(weights[t] * float(Decimal(results[t][2]["tier_score"])) for t in TIERS)
    return final, results


def report(label, pred_scores, pred_costs):
    for cap, cap_label in ((1.0, "한도최대"), (0.90, "마진90%")):
        final, res = calibrate(pred_scores, pred_costs, cap)
        detail = "  ".join(
            f"{t[:4]}:{float(Decimal(res[t][2]['tier_score'])):.4f}|{float(Decimal(res[t][2]['budget_ratio'])):.3f}"
            for t in TIERS
        )
        print(f"  {label:<14}{cap_label:<8} 최종 {final:.6f}   {detail}")


print("\n=== Dev 점수 (안전계수는 Dev에서 보정 — 공식 베이스라인과 동일 방법론) ===")
gbm_scores, gbm_costs = rows_from(dev_pred)
report("GBM", gbm_scores, gbm_costs)

repro = hash_regex.load_artifact(REPO / "build/exp/repro/artifact.json")
pred = [hash_regex.predict_episode(ep, repro) for ep in dev_in.episodes]
report("ridge(재현)", [p[0] for p in pred], [p[1] for p in pred])
