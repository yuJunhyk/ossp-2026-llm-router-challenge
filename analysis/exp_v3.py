"""T2 실험 v3: 특징 확장 + 앙상블 가중 튜닝.

- 확장 dense 특징: 문자 종류별 구성(라틴/한글/숫자/공백/기호), 줄 구조,
  코드 펜스·LaTeX 명령 수 등 — 입력 토큰(비용) 추정과 유형 판별 강화
- 변형: 기존 E(ens ridge+gbm-diff) vs 확장특징 버전, 앙상블 가중 w 그리드
- 승자는 공식 Decimal 채점기로 재검증
"""

import sys
import math
import re
from decimal import Decimal
from pathlib import Path

REPO = Path("/Users/yujunhyeog/ossp-router")
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
    _score_one_tier,
)
from ossp_router.heuristic import episode_text
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
_LATEX = re.compile(r"\\[a-zA-Z]{2,}")
_FENCE = re.compile(r"```")


def extra_dense(text):
    n = max(len(text), 1)
    latin = sum("a" <= c.lower() <= "z" for c in text)
    hangul = sum("가" <= c <= "힣" for c in text)
    digit = sum(c.isdigit() for c in text)
    space = sum(c.isspace() for c in text)
    symbol = len(text) - latin - hangul - digit - space
    lines = text.split("\n")
    max_line = max((len(l) for l in lines), key=lambda v: v, default=0)
    return (
        math.log1p(latin),
        math.log1p(hangul),
        math.log1p(digit),
        math.log1p(space),
        math.log1p(symbol),
        latin / n,
        symbol / n,
        math.log1p(len(lines)),
        math.log1p(max_line),
        math.log1p(len(_FENCE.findall(text))),
        math.log1p(len(_LATEX.findall(text))),
        math.log1p(text.count("?") + text.count("？")),
    )


X_tr_base, Y_tr = _training_matrix(train_in, train_out, policy, HASH_BINS)
X_dev_base = np.asarray(
    [hash_regex.raw_feature_vector(ep, HASH_BINS) for ep in dev_in.episodes]
)
ext_tr = np.asarray([extra_dense(episode_text(ep)) for ep in train_in.episodes])
ext_dev = np.asarray([extra_dense(episode_text(ep)) for ep in dev_in.episodes])
X_tr_ext = np.hstack([X_tr_base, ext_tr])
X_dev_ext = np.hstack([X_dev_base, ext_dev])
N = X_tr_base.shape[0]

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
DIFF_IDX = [(0, None), (1, 0), (2, 0), (3, None), (4, None), (5, None)]


def make_diff_targets(Y):
    out = Y.copy()
    out[:, 1] = Y[:, 1] - Y[:, 0]
    out[:, 2] = Y[:, 2] - Y[:, 0]
    return out


def reconstruct(raw):
    out = raw.copy()
    out[:, 1] = raw[:, 0] + raw[:, 1]
    out[:, 2] = raw[:, 0] + raw[:, 2]
    return out


def gbm_diff_fit(X, X_eval, round_grid=(50, 100, 200, 400)):
    targets = make_diff_targets(Y_tr)
    T = targets.shape[1]
    dev = np.empty((X_eval.shape[0], T))
    for t in range(T):
        best = None
        for rounds in round_grid:
            cur = np.empty(N)
            for f in range(FOLDS):
                va = fold_ids == f
                ds = lgb.Dataset(X[~va], label=targets[~va, t])
                cur[va] = lgb.train(PARAMS, ds, num_boost_round=rounds).predict(X[va])
            mse = float(np.mean((cur - targets[:, t]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, rounds)
        ds = lgb.Dataset(X, label=targets[:, t])
        dev[:, t] = lgb.train(PARAMS, ds, num_boost_round=best[1]).predict(X_eval)
    return reconstruct(dev)


def ridge_fit(X, X_eval, alpha=10.0):
    mean, scale, intercept, coef = _fit_ridge(X, Y_tr, alpha)
    return _predict_ridge(X_eval, mean, scale, intercept, coef)


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


def calibrate_fast(pred, cap_of_limit=0.90):
    ps, pc = rows_from(pred)
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
                ps, pc, budget_multiplier=mult, safety_ratio=safety
            )
            score, ratio = fast_eval(selected)
            if ratio > mult * cap_of_limit + 1e-12:
                continue
            if best is None or (score, -ratio) > (best[0], -best[1]):
                best = (score, ratio, safety)
        detail[tier] = best
        total += weights[tier] * best[0]
    return total, detail


print("모델 학습 중 ...")
ridge_base = ridge_fit(X_tr_base, X_dev_base)
ridge_ext = ridge_fit(X_tr_ext, X_dev_ext)
gbmd_base = gbm_diff_fit(X_tr_base, X_dev_base)
gbmd_ext = gbm_diff_fit(X_tr_ext, X_dev_ext)

print("\nDev 마진90% 최종점수 (빠른 채점):")
combos = {}
for label, (r, g) in {
    "기본특징": (ridge_base, gbmd_base),
    "확장특징": (ridge_ext, gbmd_ext),
}.items():
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (1 - w) * r + w * g
        total, detail = calibrate_fast(pred)
        combos[(label, w)] = (total, detail, pred)
        print(f"  {label} w(gbm)={w:.1f}: {total:.6f}")

(best_label, best_w), (best_total, best_detail, best_pred) = max(
    combos.items(), key=lambda kv: kv[1][0]
)
print(f"\n최선: {best_label} w={best_w} → {best_total:.6f}")
for tier in TIERS:
    s, r, safety = best_detail[tier]
    print(f"  {tier}: 점수 {s:.6f}  비용비율 {r:.4f}  안전계수 {safety:.4f}")

# --- 공식 Decimal 채점기로 재검증 ---
print("\n공식 채점기 재검증 (마진90% 안전계수 적용):")
ps, pc = rows_from(best_pred)
weights = {
    "fast": Decimal("0.4"),
    "balanced": Decimal("0.3"),
    "premium": Decimal("0.3"),
}
final = Decimal(0)
for tier in TIERS:
    safety = best_detail[tier][2]
    selected, _ = hash_regex.select_models(
        ps,
        pc,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=safety,
    )
    rep = _score_one_tier(dev_in, dev_out, policy, tier, selected)
    final += weights[tier] * Decimal(rep["tier_score"])
    print(
        f"  {tier}: 점수 {rep['tier_score']}  비용비율 {rep['budget_ratio']}  통과 {rep['budget_passed']}"
    )
print(f"공식 가중 최종점수: {final}")
