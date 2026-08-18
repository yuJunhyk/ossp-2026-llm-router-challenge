# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""C3 게이트 — 점수 헤드 로지스틱 교체 (train 전용, V15-GATES.md 사전 등록).

점수 라벨은 {0, 0.25, 0.5, 0.75, 1} — num_generations(2 또는 4)회 생성의 성공
비율이다. 제곱오차 회귀 대신 binomial weight = num_generations를 건 분수 타깃
로지스틱으로 점수 3헤드를 다시 적합한다. 비용 3헤드는 현행 ridge(λ=10) 유지.

8,232차원 primal은 불가하므로 dual(가중 커널) IRLS를 쓴다. fold당 K=XXᵀ를
한 번 계산해 ridge 비용 헤드와 3개 점수 타깃·전 반복에 재사용한다.

  IRLS: α ← (K + λ·diag(1/W))⁻¹ z,  W_i = n_i·p_i(1-p_i),
        z = f + n_i(y_i - p_i)/W_i,  f = Kα

λ_logit ∈ {3, 10, 30} (사전 등록 최소 격자, 최고 CV가 후보 대표값).
게이트: 동일 fold 짝지음 ΔCV ≥ +0.004. 스크리닝 전 기준선 0.6553 재현 필수.
"""

from __future__ import annotations

import json
import math
import subprocess
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
    group_fold_ids,
    linear_candidate,
    realized,
    rows_with_pessimism,
    stress_masks,
    template_group_keys,
)
from os2_features import TOTAL_DIM, extract_sparse
from os2_policy import allocate
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

LAM_COST = 10.0
LOGIT_LAMBDAS = (3.0, 10.0, 30.0)
BASELINE_CV = 0.6553
GATE_DELTA = 0.004
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}
IRLS_MAXIT = 15
IRLS_TOL = 1e-4


def _sigmoid(f):
    return 1.0 / (1.0 + np.exp(-np.clip(f, -30.0, 30.0)))


def logit_candidate(X, Y, ngen, fold_sets, lam_logit):
    """점수 3헤드 = 가중 dual IRLS 로지스틱, 비용 3헤드 = ridge(λ=10).

    linear_candidate와 동일 계약의 bundles/oof를 반환한다. K를 fold당 1회만 계산.
    """
    n = X.shape[0]
    bundles, oof_by_seed = [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            X_tr, X_va = X[tr], X[va]
            K_tr = X_tr @ X_tr.T
            K_va = X_va @ X_tr.T
            n_tr = K_tr.shape[0]

            # 비용 헤드: dual ridge (기존과 동일한 해 — K 재사용 형태)
            A = K_tr.copy()
            A[np.diag_indices(n_tr)] += LAM_COST
            alpha_cost = np.linalg.solve(A, Y[tr][:, 3:])
            pred_tr_cost = K_tr @ alpha_cost
            resid = Y[tr][:, 3:] - pred_tr_cost
            smear = np.exp(resid).mean(axis=0)
            sigma = resid.std(axis=0)

            # 점수 헤드: 가중 IRLS 로지스틱 (타깃 3개)
            score_va = np.zeros((int(va.sum()), 3))
            score_tr_oof = None  # 진단 불필요 — OOF는 va 예측만 사용
            w_trials = ngen[tr]
            for j in range(3):
                y = Y[tr][:, j]
                alpha = np.zeros(n_tr)
                f = np.zeros(n_tr)
                for _ in range(IRLS_MAXIT):
                    p = np.clip(_sigmoid(f), 1e-6, 1.0 - 1e-6)
                    Wd = np.maximum(w_trials * p * (1.0 - p), 1e-8)
                    z = f + w_trials * (y - p) / Wd
                    A = K_tr + lam_logit * np.diag(1.0 / Wd)
                    alpha = np.linalg.solve(A, z)
                    f_new = K_tr @ alpha
                    if np.max(np.abs(f_new - f)) < IRLS_TOL:
                        f = f_new
                        break
                    f = f_new
                score_va[:, j] = _sigmoid(K_va @ alpha)

            pred_va = np.hstack([score_va, K_va @ alpha_cost])
            raw_full[va] = pred_va
            smear_acc.append(smear)
            sigma_acc.append(sigma)
            bundles.append(
                {
                    "idx": np.where(va)[0],
                    "pred": pred_va,
                    "smear": smear,
                    "sigma": sigma,
                }
            )
        oof_by_seed.append(
            {
                "pred": raw_full,
                "smear": np.mean(smear_acc, axis=0),
                "sigma": np.mean(sigma_acc, axis=0),
            }
        )
    return bundles, oof_by_seed


def fold_weighted_scores(bundles, tier_cfgs, truth, policy):
    true_score, true_cost = truth["score"], truth["cost"]
    per_fold, violations = [], 0
    for b in bundles:
        total = 0.0
        for tier in TIERS:
            beta, margin = tier_cfgs[tier]
            mult = float(policy.tiers[tier].budget_multiplier)
            weight = float(policy.tiers[tier].weight)
            rows = rows_with_pessimism(b["pred"], b["smear"], b["sigma"], beta)
            choice = allocate(rows, mult, margin)
            s, u = realized(b["idx"], choice, true_score, true_cost, b["light_base"])
            if u > mult:
                violations += 1
            total += weight * s
        per_fold.append(total)
    return np.array(per_fold), violations


def main() -> int:
    policy = load_bundled_policy()
    inputs = load_input(REPO / "data/materialized/train/inputs.json")
    outcomes = load_outcomes(REPO / "data/train/outcomes.json")
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    rates = {m: policy.models[m] for m in MODEL_IDS}
    scores, tokens = {}, {}
    for oc in outcomes.outcomes:
        scores[(oc.episode_id, oc.model_id)] = float(oc.score)
        tokens[(oc.episode_id, oc.model_id)] = (oc.input_tokens, oc.output_tokens)
    eids = [ep.episode_id for ep in inputs.episodes]

    # num_generations는 protocol 객체에 없을 수 있어 원본 JSON에서 직접 읽는다
    raw = json.loads((REPO / "data/train/outcomes.json").read_text(encoding="utf-8"))
    ngen_map = {}
    for ep in raw["episodes"]:
        for m, rec in ep["models"].items():
            ngen_map[(ep["episode_id"], m)] = int(rec.get("num_generations", 1))
    # 문항 대표값: 세 모델 중 최소 (보수적 가중)
    ngen = np.array(
        [min(ngen_map.get((e, m), 1) for m in MODEL_IDS) for e in eids], dtype=float
    )
    from collections import Counter

    print(f"num_generations 분포: {dict(Counter(ngen.astype(int)))}")

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

    Y = np.zeros((n, 6))
    for i in range(n):
        for j, m in enumerate(MODEL_IDS):
            Y[i, j] = true_score[i][m]
            Y[i, 3 + j] = math.log(max(true_cost[i][m], 1e-9))

    keys = template_group_keys(texts)
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    masks = stress_masks(texts)

    def attach_light_base(bundles):
        for b in bundles:
            b["light_base"] = sum(light_costs[i] for i in b["idx"])
        return bundles

    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val

    print("[1/3] 기준선 재현 (ridge 점수 헤드, λ=10)", flush=True)
    b_base, oof_base = linear_candidate(X, Y, fold_sets, LAM_COST)
    res_base = calibrate(
        "baseline", attach_light_base(b_base), oof_base, masks, truth, policy
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효, 프로토콜 점검 필요")
        return 2

    print("[2/3] 변형 평가 (로지스틱 점수 헤드, λ_logit 격자)", flush=True)
    best = None
    grid_report = {}
    for lam_logit in LOGIT_LAMBDAS:
        b_log, oof_log = logit_candidate(X, Y, ngen, fold_sets, lam_logit)
        res = calibrate(
            f"logit(λ={lam_logit})",
            attach_light_base(b_log),
            oof_log,
            masks,
            truth,
            policy,
        )
        grid_report[str(lam_logit)] = res["weighted"]
        print(f"  λ_logit={lam_logit:>4}: weighted_cv={res['weighted']:.4f}")
        if best is None or res["weighted"] > best[1]["weighted"]:
            best = (lam_logit, res, b_log)
    lam_best, res_log, b_log = best
    delta = res_log["weighted"] - res_base["weighted"]
    print(f"  최고 λ_logit={lam_best}: Δ={delta:+.4f}")
    for tier, cfg in res_log["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )

    print("[3/3] 진단 — paired per-fold Δ + v1.3 고정 캘리브레이션", flush=True)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    log_cfg = {
        t: (res_log["tiers"][t]["beta"], res_log["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(
        attach_light_base(b_base), base_cfg, truth, policy
    )
    pf_log, _ = fold_weighted_scores(attach_light_base(b_log), log_cfg, truth, policy)
    pf_delta = pf_log - pf_base
    pos = int((pf_delta > 0).sum())
    print(f"  per-fold Δ>0: {pos}/15 (참고 지표 ≥10)")

    fx_base, vio_b = fold_weighted_scores(
        attach_light_base(b_base), V13_CONFIG, truth, policy
    )
    fx_log, vio_l = fold_weighted_scores(
        attach_light_base(b_log), V13_CONFIG, truth, policy
    )
    fixed_delta = float(fx_log.mean() - fx_base.mean())
    print(
        f"  v1.3 고정 캘리브레이션 Δ={fixed_delta:+.4f} (예산 위반 base {vio_b}, logit {vio_l})"
    )

    verdict = delta >= GATE_DELTA
    print(f"C3 게이트: {'통과' if verdict else '미달 → 기각'} (문턱 +{GATE_DELTA})")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v15/logistic-head.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate": "C3 logistic-head",
                "git": git_hash,
                "lambda_grid": grid_report,
                "lambda_best": lam_best,
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_log["weighted"],
                "delta": delta,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "fixed_v13_delta": fixed_delta,
                "pass": bool(verdict),
                "tiers": res_log["tiers"],
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
