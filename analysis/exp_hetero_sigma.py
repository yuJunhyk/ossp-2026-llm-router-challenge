# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""C4 게이트 — 비용 이분산 σ_i (train 전용, V15-GATES.md 사전 등록·조건부).

현행 β·σ 비관은 모델별 상수 σ를 전 문항에 동일 적용한다. 여기서는 문항별
불확실성 σ_i를 예측해 exp(β·σ_i)로 차등 비관한다. 예산 방어를 약화시키지
않기 위한 세 가지 장치(V15-GATES.md):

  1. 선행 관문 — calibrate fork가 σ_i≡상수일 때 원본 calibrate와 완전 동일한
     결과(0.6553)를 내야 한다. 미통과면 본 실험 무효.
  2. 정직 잔차 — σ_i 모델은 fold-train 내부 그룹 3-fold의 held-out 잔차로만
     학습한다 (학습 잔차 사용 시 σ 하향 편향 → 비관 버퍼 축소 → 파국 모드).
  3. β 격자 의미 유지 — fold-train에서 mean(σ_i) = pooled σ가 되도록 정규화,
     σ_i는 [0.25, 3.0]×pooled로 클립.

고정값(등록): λ_sigma=10, 내부 fold 3(그룹 기반, seed 7), 클립 [0.25, 3.0].
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
    BETAS,
    CV_SEEDS,
    FOLDS,
    MARGINS,
    SAFETY_CAP,
    STRESS_CAP,
    calibrate,
    cost_stats,
    episode_text,
    fit_dual_ridge,
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

LAM = 10.0
LAM_SIGMA = 10.0
INNER_FOLDS = 3
INNER_SEED = 7
SIGMA_CLIP = (0.25, 3.0)
BASELINE_CV = 0.6553
GATE_DELTA = 0.004
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}


def rows_with_pessimism_ep(pred, smear, sigma_ep, beta):
    """rows_with_pessimism의 문항별 σ 버전. sigma_ep는 (n,3), light 열은 비관 제외."""
    scores = np.clip(pred[:, :3], 0.0, 1.0)
    cost = np.exp(np.clip(pred[:, 3:], -20.0, 5.0)) * smear[None, :]
    pess = np.ones_like(cost)
    pess[:, 1] = np.exp(beta * sigma_ep[:, 1])
    pess[:, 2] = np.exp(beta * sigma_ep[:, 2])
    cost = cost * pess
    cost[:, 0] = np.maximum(cost[:, 0], 1e-12)
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * 1.01)
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * 1.01)
    out = []
    for i in range(pred.shape[0]):
        out.append(
            {
                MODEL_IDS[0]: (float(scores[i, 0]), float(cost[i, 0])),
                MODEL_IDS[1]: (float(scores[i, 1]), float(cost[i, 1])),
                MODEL_IDS[2]: (float(scores[i, 2]), float(cost[i, 2])),
            }
        )
    return out


def calibrate_ep(name, bundles, oof_by_seed, masks, truth, policy):
    """calibrate의 fork — b["sigma_ep"]/o["sigma_ep"]를 쓰는 것 외에는 동일."""
    true_score, true_cost = truth["score"], truth["cost"]
    light_costs = truth["light_costs"]
    result = {"name": name, "tiers": {}, "weighted": 0.0}
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        weight = float(policy.tiers[tier].weight)
        feasible = []
        for beta in BETAS:
            fold_rows = [
                rows_with_pessimism_ep(b["pred"], b["smear"], b["sigma_ep"], beta)
                for b in bundles
            ]
            oof_rows = [
                rows_with_pessimism_ep(o["pred"], o["smear"], o["sigma_ep"], beta)
                for o in oof_by_seed
            ]
            for margin in MARGINS:
                scores, useds = [], []
                for b, rows in zip(bundles, fold_rows):
                    choice = allocate(rows, mult, margin)
                    s, u = realized(
                        b["idx"], choice, true_score, true_cost, b["light_base"]
                    )
                    scores.append(s)
                    useds.append(u)
                if max(useds) > mult * SAFETY_CAP[tier]:
                    continue
                stress_ok, stress_worst = True, 0.0
                for mask_idx in masks.values():
                    base = sum(light_costs[i] for i in mask_idx)
                    for rows in oof_rows:
                        sub = [rows[i] for i in mask_idx]
                        choice = allocate(sub, mult, margin)
                        _, u = realized(mask_idx, choice, true_score, true_cost, base)
                        stress_worst = max(stress_worst, u)
                        if u > mult * STRESS_CAP:
                            stress_ok = False
                            break
                    if not stress_ok:
                        break
                if stress_ok:
                    feasible.append(
                        (
                            float(np.mean(scores)),
                            -max(useds),
                            beta,
                            margin,
                            stress_worst,
                        )
                    )
        if not feasible:
            result["tiers"][tier] = None
            continue
        feasible.sort(reverse=True)
        mean_s, neg_u, beta, margin, stress_worst = feasible[0]
        result["tiers"][tier] = {
            "beta": beta,
            "margin": margin,
            "cv_score": mean_s,
            "max_used_15fold": -neg_u,
            "budget_multiplier": mult,
            "stress_worst": stress_worst,
        }
        result["weighted"] += weight * mean_s
    if any(v is None for v in result["tiers"].values()):
        result["weighted"] = float("-inf")
    return result


def hetero_candidate(X, Y, fold_sets, keys, lam):
    """점수·비용 헤드는 현행 ridge 그대로, σ만 문항별로 — sigma_ep 포함 번들 반환."""
    n = X.shape[0]
    bundles, oof_by_seed = [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        sig_full = np.zeros((n, 3))
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            tr_idx = np.where(tr)[0]
            W = fit_dual_ridge(X[tr], Y[tr], lam)
            pred_tr = X[tr] @ W
            smear, sigma_pool = cost_stats(pred_tr[:, 3:], Y[tr][:, 3:])
            pred_va = X[va] @ W

            # 정직 잔차: fold-train 내부 그룹 3-fold held-out 잔차
            keys_tr = [keys[i] for i in tr_idx]
            inner = group_fold_ids(keys_tr, INNER_SEED, folds=INNER_FOLDS)
            r_abs = np.zeros((len(tr_idx), 3))
            for j in range(INNER_FOLDS):
                iva = inner == j
                itr = ~iva
                Wi = fit_dual_ridge(X[tr][itr], Y[tr][itr], lam)
                pv = X[tr][iva] @ Wi
                r_abs[iva] = np.abs(Y[tr][iva][:, 3:] - pv[:, 3:])

            # σ_i 모델: X → |정직 잔차|, 정규화 mean(σ_i|tr) = pooled σ, 클립
            Ws = fit_dual_ridge(X[tr], r_abs, LAM_SIGMA)
            r_hat_tr = np.maximum(X[tr] @ Ws, 1e-6)
            r_hat_va = np.maximum(X[va] @ Ws, 1e-6)
            scale = sigma_pool / r_hat_tr.mean(axis=0)
            sig_va = np.clip(
                r_hat_va * scale[None, :],
                SIGMA_CLIP[0] * sigma_pool[None, :],
                SIGMA_CLIP[1] * sigma_pool[None, :],
            )

            raw_full[va] = pred_va
            sig_full[va] = sig_va
            smear_acc.append(smear)
            sigma_acc.append(sigma_pool)
            bundles.append(
                {
                    "idx": np.where(va)[0],
                    "pred": pred_va,
                    "smear": smear,
                    "sigma": sigma_pool,
                    "sigma_ep": sig_va,
                }
            )
        oof_by_seed.append(
            {
                "pred": raw_full,
                "smear": np.mean(smear_acc, axis=0),
                "sigma": np.mean(sigma_acc, axis=0),
                "sigma_ep": sig_full,
            }
        )
    return bundles, oof_by_seed


def fold_weighted_scores_ep(bundles, tier_cfgs, truth, policy):
    true_score, true_cost = truth["score"], truth["cost"]
    per_fold, violations = [], 0
    for b in bundles:
        total = 0.0
        for tier in TIERS:
            beta, margin = tier_cfgs[tier]
            mult = float(policy.tiers[tier].budget_multiplier)
            weight = float(policy.tiers[tier].weight)
            rows = rows_with_pessimism_ep(b["pred"], b["smear"], b["sigma_ep"], beta)
            choice = allocate(rows, mult, margin)
            s, u = realized(b["idx"], choice, true_score, true_cost, b["light_base"])
            if u > mult:
                violations += 1
            total += weight * s
        per_fold.append(total)
    return np.array(per_fold), violations


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

    print("[1/4] 기준선 재현 (λ=10)", flush=True)
    b_base, oof_base = linear_candidate(X, Y, fold_sets, LAM)
    res_base = calibrate(
        "baseline", attach_light_base(b_base), oof_base, masks, truth, policy
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효")
        return 2

    print("[2/4] 선행 관문 — calibrate fork 동등성 (σ_i≡상수)", flush=True)
    for b in b_base:
        b["sigma_ep"] = np.tile(b["sigma"], (b["pred"].shape[0], 1))
    for o in oof_base:
        o["sigma_ep"] = np.tile(o["sigma"], (o["pred"].shape[0], 1))
    res_eq = calibrate_ep("fork-eq", b_base, oof_base, masks, truth, policy)
    eq_diff = abs(res_eq["weighted"] - res_base["weighted"])
    print(f"  fork weighted_cv={res_eq['weighted']:.4f}  |diff|={eq_diff:.2e}")
    if eq_diff > 1e-9:
        print("  선행 관문 실패 — 본 실험 무효 (V15-GATES.md)")
        return 3

    print("[3/4] 변형 평가 (문항별 σ_i, 정직 잔차)", flush=True)
    b_het, oof_het = hetero_candidate(X, Y, fold_sets, keys, LAM)
    res_het = calibrate_ep(
        "hetero-sigma", attach_light_base(b_het), oof_het, masks, truth, policy
    )
    delta = res_het["weighted"] - res_base["weighted"]
    print(f"  hetero weighted_cv={res_het['weighted']:.4f}  Δ={delta:+.4f}")
    for tier, cfg in res_het["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )

    print("[4/4] 진단", flush=True)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    het_cfg = {
        t: (res_het["tiers"][t]["beta"], res_het["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(b_base, base_cfg, truth, policy)
    pf_het, _ = fold_weighted_scores_ep(
        attach_light_base(b_het), het_cfg, truth, policy
    )
    pf_delta = pf_het - pf_base
    pos = int((pf_delta > 0).sum())
    print(f"  per-fold Δ>0: {pos}/15")

    fx_base, vio_b = fold_weighted_scores(b_base, V13_CONFIG, truth, policy)
    fx_het, vio_h = fold_weighted_scores_ep(
        attach_light_base(b_het), V13_CONFIG, truth, policy
    )
    fixed_delta = float(fx_het.mean() - fx_base.mean())
    print(
        f"  v1.3 고정 캘리브레이션 Δ={fixed_delta:+.4f} (예산 위반 base {vio_b}, het {vio_h})"
    )

    verdict = delta >= GATE_DELTA
    print(f"C4 게이트: {'통과' if verdict else '미달 → 기각'} (문턱 +{GATE_DELTA})")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v15/hetero-sigma.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate": "C4 hetero-sigma",
                "git": git_hash,
                "fork_equivalence_diff": eq_diff,
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_het["weighted"],
                "delta": delta,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "fixed_v13_delta": fixed_delta,
                "pass": bool(verdict),
                "tiers": res_het["tiers"],
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
