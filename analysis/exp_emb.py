# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""임베딩 라우터 게이트 실험.

1) train OOF에서 임베딩 예측기(kNN·ridge)와 기존 ens의 결합을 비교·선택
2) 승자 구성으로 Dev 게이트 판정 1회: v1.2(표면특징+fast버퍼1.16) vs v2(+임베딩)

Dev는 이 스크립트의 마지막 게이트 평가에서만 사용한다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

import numpy as np
import lightgbm as lgb

import hash_regex
from train_hash_regex import _fit_ridge, _predict_ridge, _training_matrix
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

policy = load_bundled_policy()
train_in = load_input(REPO / "data/materialized/train/inputs.json")
train_out = load_outcomes(REPO / "data/train/outcomes.json")
dev_in = load_input(REPO / "data/materialized/dev/inputs.json")
dev_out = load_outcomes(REPO / "data/dev/outcomes.json")

X_tr, Y_tr = _training_matrix(train_in, train_out, policy, 256)
X_dev = np.asarray([hash_regex.raw_feature_vector(ep, 256) for ep in dev_in.episodes])
N = len(X_tr)
FOLDS = 5
fold_ids = np.arange(N) % FOLDS
DIFF = Y_tr.copy()
DIFF[:, 1] = Y_tr[:, 1] - Y_tr[:, 0]
DIFF[:, 2] = Y_tr[:, 2] - Y_tr[:, 0]
GBM_PARAMS = dict(
    objective="regression",
    num_leaves=7,
    learning_rate=0.03,
    min_data_in_leaf=15,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
)
GBM_ROUNDS = [200, 100, 200, 800, 800, 800]  # train_ens에서 OOF로 확정된 값
SEEDS = (7, 8, 9)
BUFFERS = {"fast": 1.16, "balanced": 1.10, "premium": 1.30}
WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def reconstruct(p):
    out = p.copy()
    out[:, 1] = p[:, 0] + p[:, 1]
    out[:, 2] = p[:, 0] + p[:, 2]
    return out


def actuals(outcomes, inputs):
    truth = {}
    for oc in outcomes.outcomes:
        r = policy.models[oc.model_id]
        cost = (
            float(r.fixed_cost)
            + (
                oc.input_tokens * float(r.input_token_rate)
                + oc.output_tokens * float(r.output_token_rate)
            )
            / policy.token_unit
        )
        truth[(oc.episode_id, oc.model_id)] = (float(oc.score), cost)
    ids = [ep.episode_id for ep in inputs.episodes]
    s = {m: np.array([truth[(i, m)][0] for i in ids]) for m in MODEL_IDS}
    c = {m: np.array([truth[(i, m)][1] for i in ids]) for m in MODEL_IDS}
    return s, c


tr_s, tr_c = actuals(train_out, train_in)
dev_s, dev_c = actuals(dev_out, dev_in)


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


def largest_safe(pred, tier, true_c):
    """train 실제 비용으로 안전계수 탐색 (배포 버퍼 반영)."""
    mult = float(policy.tiers[tier].budget_multiplier)
    scores, costs = rows_from(pred)
    light_total = float(sum(true_c["ax31-light"]))
    lo = 1.0 / mult
    for i in range(120, -1, -1):
        safety = lo + (1 - lo) * i / 120
        selected, _ = hash_regex.select_models(
            scores, costs, budget_multiplier=mult, safety_ratio=safety
        )
        actual = sum(true_c[m][j] for j, m in enumerate(selected)) / light_total
        if actual * BUFFERS[tier] <= mult:
            return safety
    return lo


def simulated_final(pred, true_s, true_c, safeties):
    """주어진 안전계수로 선택했을 때 실제 가중 점수·비용비율."""
    scores, costs = rows_from(pred)
    light_total = float(sum(true_c["ax31-light"]))
    total = 0.0
    detail = {}
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        selected, _ = hash_regex.select_models(
            scores, costs, budget_multiplier=mult, safety_ratio=safeties[tier]
        )
        s = float(np.mean([true_s[m][j] for j, m in enumerate(selected)]))
        ratio = sum(true_c[m][j] for j, m in enumerate(selected)) / light_total
        passed = ratio <= mult
        total += WEIGHTS[tier] * (s if passed else 0.0)
        detail[tier] = (s, float(ratio), passed)
    return total, detail


def uplift_corr(pred):
    tu_m = Y_tr[:, 1] - Y_tr[:, 0]
    tu_k = Y_tr[:, 2] - Y_tr[:, 0]
    pu_m = pred[:, 1] - pred[:, 0]
    pu_k = pred[:, 2] - pred[:, 0]
    return float(np.corrcoef(pu_m, tu_m)[0, 1]), float(np.corrcoef(pu_k, tu_k)[0, 1])


# ---------- 기존 ens (표면특징) OOF ----------
print("[1] 기존 ens OOF 재계산")
ridge_oof = np.empty_like(Y_tr)
for f in range(FOLDS):
    va = fold_ids == f
    m, s, b, c = _fit_ridge(X_tr[~va], Y_tr[~va], 10.0)
    ridge_oof[va] = _predict_ridge(X_tr[va], m, s, b, c)
gbm_oof = np.empty_like(DIFF)
for t in range(6):
    for f in range(FOLDS):
        va = fold_ids == f
        ds = lgb.Dataset(X_tr[~va], label=DIFF[~va, t])
        gbm_oof[va, t] = lgb.train(
            dict(GBM_PARAMS, seed=7), ds, num_boost_round=GBM_ROUNDS[t]
        ).predict(X_tr[va])
base_oof = 0.5 * ridge_oof + 0.5 * reconstruct(gbm_oof)
cm, ck = uplift_corr(base_oof)
print(f"  base ens: corr(uplift_M)={cm:+.3f} corr(uplift_K)={ck:+.3f}")


# ---------- 임베딩 예측기 OOF ----------
def knn_oof(E, k):
    pred = np.empty_like(DIFF)
    for f in range(FOLDS):
        va = fold_ids == f
        sims = E[va] @ E[~va].T  # 코사인 (정규화 완료)
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        w = np.take_along_axis(sims, top, axis=1)
        w = np.clip(w, 0, None) + 1e-9
        w = w / w.sum(axis=1, keepdims=True)
        tgt = DIFF[~va]
        pred[va] = np.einsum("nk,nkt->nt", w, tgt[top])
    return reconstruct(pred)


def ridge_emb_oof(E, alpha):
    pred = np.empty_like(DIFF)
    for f in range(FOLDS):
        va = fold_ids == f
        m, s, b, c = _fit_ridge(E[~va], DIFF[~va], alpha)
        pred[va] = _predict_ridge(E[va], m, s, b, c)
    return reconstruct(pred)


emb_oof = {}
for key in ("e5small", "minilm"):
    E = np.load(REPO / f"build/embeddings/{key}-train.npy").astype(np.float64)
    print(f"[2] {key} (차원 {E.shape[1]})")
    for k in (8, 16, 32):
        p = knn_oof(E, k)
        cm, ck = uplift_corr(p)
        emb_oof[(key, f"knn{k}")] = p
        print(f"  knn k={k:<3} corr(uplift_M)={cm:+.3f} corr(uplift_K)={ck:+.3f}")
    for alpha in (1.0, 10.0, 100.0):
        p = ridge_emb_oof(E, alpha)
        cm, ck = uplift_corr(p)
        emb_oof[(key, f"ridge{alpha:g}")] = p
        print(
            f"  ridge α={alpha:<5g} corr(uplift_M)={cm:+.3f} corr(uplift_K)={ck:+.3f}"
        )

# 결합 후보: base와 최고 임베딩 예측기들의 가중 평균
best_emb_key = max(emb_oof, key=lambda kk: sum(uplift_corr(emb_oof[kk])))
print(f"\n[3] 최고 임베딩 예측기: {best_emb_key}")
combos = {"v1.2(base)": base_oof}
for w in (0.3, 0.5, 0.7):
    combos[f"base+{best_emb_key[1]}@{w}"] = (1 - w) * base_oof + w * emb_oof[
        best_emb_key
    ]

print(f"\n[4] train 모의 최종점수 (안전계수는 각 구성의 OOF로 산정, 버퍼 {BUFFERS})")
results = {}
for name, pred in combos.items():
    safeties = {tier: largest_safe(pred, tier, tr_c) for tier in TIERS}
    total, detail = simulated_final(pred, tr_s, tr_c, safeties)
    cm, ck = uplift_corr(pred)
    results[name] = (total, safeties)
    print(f"  {name:<24} train모의 {total:.6f}  corrM {cm:+.3f}  corrK {ck:+.3f}")

winner = max(results, key=lambda n: results[n][0])
print(f"\n[5] OOF 승자: {winner} → Dev 게이트 판정 (유일한 Dev 사용)")


# ---------- Dev 게이트: v1.2 vs 승자 — 전체 train 학습 후 Dev 예측 ----------
def full_predictions(with_emb_weight=0.0, emb_key=None, emb_kind=None):
    m, s, b, c = _fit_ridge(X_tr, Y_tr, 10.0)
    ridge_dev = _predict_ridge(X_dev, m, s, b, c)
    gbm_dev = np.zeros((len(X_dev), 6))
    for t in range(6):
        for seed in SEEDS:
            ds = lgb.Dataset(X_tr, label=DIFF[:, t])
            gbm_dev[:, t] += lgb.train(
                dict(GBM_PARAMS, seed=seed), ds, num_boost_round=GBM_ROUNDS[t]
            ).predict(X_dev) / len(SEEDS)
    base_dev = 0.5 * ridge_dev + 0.5 * reconstruct(gbm_dev)
    if with_emb_weight == 0.0:
        return base_dev
    E_tr = np.load(REPO / f"build/embeddings/{emb_key}-train.npy").astype(np.float64)
    E_dev = np.load(REPO / f"build/embeddings/{emb_key}-dev.npy").astype(np.float64)
    if emb_kind.startswith("knn"):
        k = int(emb_kind[3:])
        sims = E_dev @ E_tr.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        w = np.take_along_axis(sims, top, axis=1)
        w = np.clip(w, 0, None) + 1e-9
        w = w / w.sum(axis=1, keepdims=True)
        emb_dev = reconstruct(np.einsum("nk,nkt->nt", w, DIFF[top]))
    else:
        alpha = float(emb_kind[5:])
        m2, s2, b2, c2 = _fit_ridge(E_tr, DIFF, alpha)
        emb_dev = reconstruct(_predict_ridge(E_dev, m2, s2, b2, c2))
    return (1 - with_emb_weight) * base_dev + with_emb_weight * emb_dev


gate = {}
for name in ("v1.2(base)", winner):
    if name == "v1.2(base)":
        dev_pred = full_predictions()
    else:
        w = float(name.split("@")[1])
        dev_pred = full_predictions(w, best_emb_key[0], best_emb_key[1])
    safeties = results[name][1]
    total, detail = simulated_final(dev_pred, dev_s, dev_c, safeties)
    gate[name] = total
    print(f"\n  {name}: Dev 최종 {total:.6f}")
    for tier in TIERS:
        s, ratio, passed = detail[tier]
        mult = float(policy.tiers[tier].budget_multiplier)
        print(
            f"    {tier}: 점수 {s:.4f}  비용비율 {ratio:.4f}/{mult} ({ratio/mult*100:.1f}%)  통과 {passed}"
        )

if winner != "v1.2(base)":
    delta = gate[winner] - gate["v1.2(base)"]
    print(
        f"\n게이트 판정: 개선 {delta:+.6f} → {'통과 (런타임 통합 진행)' if delta >= 0.008 else '미달 (v1.2 동결)'}"
    )
else:
    print("\n게이트 판정: 임베딩 결합이 train 모의에서조차 못 이김 → v1.2 동결")
