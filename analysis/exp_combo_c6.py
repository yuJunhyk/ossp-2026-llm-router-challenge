# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""C6 게이트 — C1×C3 조합: 템플릿 유사도 특징 + 로지스틱 점수 헤드 (train 전용).

V15-GATES.md 추가 등록(2026-08-18)에 따른 탐색 실험. 구성은 고정이다:
  - 특징: C1 그대로 (fold-train 접두사 은행 유사도 3종, 슬롯 34~36)
  - 점수 헤드: C3 승자 그대로 (가중 dual IRLS 로지스틱, λ_logit=30 고정)
  - 비용 헤드: 현행 ridge(λ=10)
채택 조건: 스크리닝 ΔCV ≥ +0.004 그리고 통합 게이트 0.6613 이상.
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
from exp_template_sim import SLOT, PrefixBank, norm_prefix
from exp_logistic_head import _sigmoid
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
LAM_LOGIT = 30.0  # C3 승자 고정 — 새 격자 없음 (V15-GATES.md)
BASELINE_CV = 0.6553
GATE_DELTA = 0.004
INTEGRATED_GATE = 0.6613
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}
IRLS_MAXIT = 15
IRLS_TOL = 1e-4


def combo_candidate(X_base, Y, ngen, fold_sets, prefixes):
    """C1 특징 + C3 로지스틱 헤드 — linear_candidate 계약의 bundles/oof."""
    n = X_base.shape[0]
    bundles, oof_by_seed = [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            # C1: fold-train 은행 유사도 특징
            bank = PrefixBank([prefixes[i] for i in np.where(tr)[0]])
            X_var = X_base.copy()
            for i in range(n):
                f1, f2, f3 = bank.features(prefixes[i], exclude_self=bool(tr[i]))
                X_var[i, SLOT] = f1
                X_var[i, SLOT + 1] = f2
                X_var[i, SLOT + 2] = f3
            X_tr, X_va = X_var[tr], X_var[va]
            K_tr = X_tr @ X_tr.T
            K_va = X_va @ X_tr.T
            n_tr = K_tr.shape[0]

            # 비용 헤드: dual ridge
            A = K_tr.copy()
            A[np.diag_indices(n_tr)] += LAM_COST
            alpha_cost = np.linalg.solve(A, Y[tr][:, 3:])
            resid = Y[tr][:, 3:] - K_tr @ alpha_cost
            smear = np.exp(resid).mean(axis=0)
            sigma = resid.std(axis=0)

            # C3: 점수 헤드 로지스틱
            score_va = np.zeros((int(va.sum()), 3))
            w_trials = ngen[tr]
            for j in range(3):
                y = Y[tr][:, j]
                alpha = np.zeros(n_tr)
                f = np.zeros(n_tr)
                for _ in range(IRLS_MAXIT):
                    p = np.clip(_sigmoid(f), 1e-6, 1.0 - 1e-6)
                    Wd = np.maximum(w_trials * p * (1.0 - p), 1e-8)
                    z = f + w_trials * (y - p) / Wd
                    A = K_tr + LAM_LOGIT * np.diag(1.0 / Wd)
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

    raw = json.loads((REPO / "data/train/outcomes.json").read_text(encoding="utf-8"))
    ngen_map = {}
    for ep in raw["episodes"]:
        for m, rec in ep["models"].items():
            ngen_map[(ep["episode_id"], m)] = int(rec.get("num_generations", 1))
    ngen = np.array(
        [min(ngen_map.get((e, m), 1) for m in MODEL_IDS) for e in eids], dtype=float
    )

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
    prefixes = [norm_prefix(t) for t in texts]

    def attach_light_base(bundles):
        for b in bundles:
            b["light_base"] = sum(light_costs[i] for i in b["idx"])
        return bundles

    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val

    print("[1/3] 기준선 재현 (λ=10)", flush=True)
    b_base, oof_base = linear_candidate(X, Y, fold_sets, LAM_COST)
    res_base = calibrate(
        "baseline", attach_light_base(b_base), oof_base, masks, truth, policy
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효")
        return 2

    print("[2/3] 조합 평가 (템플릿 유사도 + 로지스틱 λ=30)", flush=True)
    b_c6, oof_c6 = combo_candidate(X, Y, ngen, fold_sets, prefixes)
    res_c6 = calibrate(
        "combo(sim+logit30)", attach_light_base(b_c6), oof_c6, masks, truth, policy
    )
    delta = res_c6["weighted"] - res_base["weighted"]
    print(f"  combo weighted_cv={res_c6['weighted']:.4f}  Δ={delta:+.4f}")
    for tier, cfg in res_c6["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )

    print("[3/3] 진단", flush=True)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    c6_cfg = {
        t: (res_c6["tiers"][t]["beta"], res_c6["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(
        attach_light_base(b_base), base_cfg, truth, policy
    )
    pf_c6, _ = fold_weighted_scores(attach_light_base(b_c6), c6_cfg, truth, policy)
    pf_delta = pf_c6 - pf_base
    pos = int((pf_delta > 0).sum())
    print(f"  per-fold Δ>0: {pos}/15")

    fx_base, vio_b = fold_weighted_scores(
        attach_light_base(b_base), V13_CONFIG, truth, policy
    )
    fx_c6, vio_c = fold_weighted_scores(
        attach_light_base(b_c6), V13_CONFIG, truth, policy
    )
    fixed_delta = float(fx_c6.mean() - fx_base.mean())
    print(
        f"  v1.3 고정 캘리브레이션 Δ={fixed_delta:+.4f} (예산 위반 base {vio_b}, c6 {vio_c})"
    )

    verdict = delta >= GATE_DELTA and res_c6["weighted"] >= INTEGRATED_GATE
    print(
        f"C6 게이트: {'통과' if verdict else '미달 → 기각'} "
        f"(스크리닝 +{GATE_DELTA} 및 통합 {INTEGRATED_GATE})"
    )

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v15/combo-c6.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate": "C6 combo sim+logit30",
                "git": git_hash,
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_c6["weighted"],
                "delta": delta,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "fixed_v13_delta": fixed_delta,
                "pass": bool(verdict),
                "tiers": res_c6["tiers"],
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
