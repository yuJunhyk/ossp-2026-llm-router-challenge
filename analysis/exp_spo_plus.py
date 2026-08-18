# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""B 게이트 — 결정 정렬 손실(SPO+) 점수 헤드 미세조정 (train 전용, V16-GATES.md).

지금까지의 모든 후보는 예측 오차를 줄이려 했다. 여기서는 배낭 결정의 regret을
SPO+ 볼록 대리손실로 직접 최소화한다 (Elmachtoub & Grigas 2022, Mandi et al. 2020).
라그랑주 완화로 tier별 목적이 문항별 3택 argmax로 분해되므로 오라클은 O(3n)이다.

설계 핵심 (설계 검증 에이전트 확정):
  - λ_t는 fold-train의 참 (점수, 비용)에 대한 이분 탐색 그림자가격. 목표 지출비는
    v1.3 고정 (β,margin)으로 그 fold-train에서 baseline이 실제 쓴 비율 — 실행
    가능한 타깃 보장. λ 지터 {0.8, 1.0, 1.25}로 컷라인 과적합 방지 (정규화 장치).
  - 파라미터화는 dual 신뢰영역: Δ = X_trᵀA, 매 스텝 RMS(K_tr A) ≤ τ 투영.
    τ=0.01 단독 판정 (0.005·0.02는 민감도 보고 전용, 후보 자격 없음). T=200 고정,
    후반 100스텝 Polyak 평균. η는 초기 기울기로 정규화 — 튜닝 대상 아님.
  - 점수 3열만 조정. 비용 3열·smear·σ는 원본 그대로 → 예산 안전 사슬 무변경.
  - 이중 게이트: calibrate ΔCV ≥ +0.004 그리고 v1.3 고정 캘리브레이션에서
    Δ ≥ 0 및 15-fold 예산 위반 0건 (캘리브레이션 이동분 차단 — C1의 교훈).

킬 스위치 (사전 등록): seed 11 fold셋으로 예비 학습 후 OOF corr(예측 3.1 uplift,
참값)의 개선이 +0.02 미만이면 본 실험 중단. 사전 실측상 3.1 uplift의 선형 상관
상한이 ~0.03(직접 회귀 λ 전수 0.009~0.025)이라 게이트 도달에 Δcorr≈+0.06이
필요하기 때문이다. 실패해도 "결정 정렬 축의 신호 부재"라는 진단이 남는다.

누수 자가검사: 그림자가격·오라클 타깃·SGD가 소비하는 모든 배열은 fold-train
슬라이스만 받도록 함수 시그니처로 강제하고, 길이를 assert로 확인한다.
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
    cost_stats,
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

LAM_RIDGE = 10.0
TAU_GATE = 0.01
TAU_SENS = (0.005, 0.02)  # 민감도 보고 전용
STEPS = 200
LAM_JITTER = (0.8, 1.0, 1.25)
BASELINE_CV = 0.6553
GATE_DELTA = 0.004
KILL_DCORR = 0.02
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}


def pess_cost(pred_log, smear, sigma, beta):
    """rows_with_pessimism의 비용 경로 재현 (배포와 동일한 ĉ)."""
    cost = np.exp(np.clip(pred_log, -20.0, 5.0)) * smear[None, :]
    pess = np.array([1.0, math.exp(beta * sigma[1]), math.exp(beta * sigma[2])])
    cost = cost * pess[None, :]
    cost[:, 0] = np.maximum(cost[:, 0], 1e-12)
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * 1.01)
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * 1.01)
    return cost


def shadow_price(S_tr, C_tr, target_ratio):
    """참 (점수,비용)에서 목표 지출비를 맞추는 λ 이분 탐색 (fold-train 전용)."""
    light = C_tr[:, 0].sum()
    lo, hi = 0.0, 1e7
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        z = np.argmax(S_tr - mid * C_tr, axis=1)
        c = C_tr[np.arange(len(z)), z].sum() / light
        if c <= target_ratio:
            hi = mid
        else:
            lo = mid
    return hi


def baseline_realized_ratio(f_tr, smear, sigma, C_tr, tier, policy):
    """v1.3 고정 설정으로 fold-train에서 baseline이 실제 쓰는 지출비 (참 비용 기준)."""
    beta, margin = V13_CONFIG[tier]
    mult = float(policy.tiers[tier].budget_multiplier)
    rows = rows_with_pessimism(f_tr, smear, sigma, beta)
    choice = allocate(rows, mult, margin)
    midx = {m: j for j, m in enumerate(MODEL_IDS)}
    spent = sum(C_tr[i, midx[c]] for i, c in enumerate(choice))
    return spent / C_tr[:, 0].sum()


def spo_train_fold(K_tr, f_tr, S_tr, C_tr, smear, sigma, policy, tau):
    """한 fold의 SPO+ 미세조정. 모든 입력은 fold-train 슬라이스만 — 누수 자가검사."""
    n_tr = K_tr.shape[0]
    assert (
        f_tr.shape[0] == S_tr.shape[0] == C_tr.shape[0] == n_tr
    ), "누수 검사: 크기 불일치"
    tier_w = {t: float(policy.tiers[t].weight) for t in TIERS}

    spec = {}
    for tier in TIERS:
        beta, _ = V13_CONFIG[tier]
        chat = pess_cost(f_tr[:, 3:], smear, sigma, beta)
        ratio = baseline_realized_ratio(f_tr, smear, sigma, C_tr, tier, policy)
        lam0 = shadow_price(S_tr, C_tr, ratio)
        lams = [g * lam0 for g in LAM_JITTER]
        jstars = [np.argmax(S_tr - l * C_tr, axis=1) for l in lams]
        spec[tier] = (chat, lams, jstars, lam0, ratio)

    rows = np.arange(n_tr)

    def grad_at(A):
        s_hat = np.clip(f_tr[:, :3] + K_tr @ A, 0.0, 1.0)
        G = np.zeros_like(s_hat)
        for tier, (chat, lams, jstars, _, _) in spec.items():
            w = tier_w[tier] * 2.0 / len(lams)
            for lam, jstar in zip(lams, jstars):
                v = s_hat - lam * chat
                u = S_tr - lam * C_tr
                jhat = np.argmax(2.0 * v - u, axis=1)
                np.add.at(G, (rows, jhat), w)
                np.add.at(G, (rows, jstar), -w)
        G[(s_hat <= 0.0) & (G > 0)] = 0.0
        G[(s_hat >= 1.0) & (G < 0)] = 0.0
        return G

    G0 = grad_at(np.zeros((n_tr, 3)))
    step_pred = K_tr @ (G0 / n_tr)
    rms0 = math.sqrt(float(np.mean(step_pred**2)))
    eta = tau / (STEPS * max(rms0, 1e-12))

    A = np.zeros((n_tr, 3))
    acc = []
    hit_boundary = 0
    for t in range(STEPS):
        A = A - eta * grad_at(A) / n_tr
        drift = K_tr @ A
        rms = math.sqrt(float(np.mean(drift**2)))
        if rms > tau:
            A *= tau / rms
            hit_boundary += 1
        if t >= STEPS // 2:
            acc.append(A.copy())
    Abar = np.mean(acc, axis=0)
    used_rms = math.sqrt(float(np.mean((K_tr @ Abar) ** 2)))
    diag = {
        "lam0": {t: spec[t][3] for t in TIERS},
        "target_ratio": {t: spec[t][4] for t in TIERS},
        "trust_region_hits": hit_boundary,
        "used_rms_vs_tau": used_rms / tau,
    }
    return Abar, diag


def spo_candidate(X, Y, S, C, fold_sets, policy, tau):
    bundles, oof_by_seed, fold_diags = [], [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            X_tr, X_va = X[tr], X[va]
            K_tr = X_tr @ X_tr.T
            K_va = X_va @ X_tr.T
            A0 = np.linalg.solve(K_tr + LAM_RIDGE * np.eye(K_tr.shape[0]), Y[tr])
            f_tr = K_tr @ A0
            smear, sigma = cost_stats(f_tr[:, 3:], Y[tr][:, 3:])
            Abar, diag = spo_train_fold(
                K_tr, f_tr, S[tr], C[tr], smear, sigma, policy, tau
            )
            fold_diags.append(diag)
            pred_va = K_va @ A0
            pred_va[:, :3] = pred_va[:, :3] + K_va @ Abar
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
    return bundles, oof_by_seed, fold_diags


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

    S = np.array([[true_score[i][m] for m in MODEL_IDS] for i in range(n)])
    C = np.array([[true_cost[i][m] for m in MODEL_IDS] for i in range(n)])
    Y = np.zeros((n, 6))
    Y[:, :3] = S
    Y[:, 3:] = np.log(np.maximum(C, 1e-9))

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
    b_base, oof_base = linear_candidate(X, Y, fold_sets, LAM_RIDGE)
    res_base = calibrate(
        "baseline", attach_light_base(b_base), oof_base, masks, truth, policy
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효")
        return 2

    print("[2/4] 킬 스위치 — seed 11 예비 학습, OOF 3.1-uplift 상관 개선", flush=True)
    kb, ko, _ = spo_candidate(X, Y, S, C, fold_sets[:1], policy, TAU_GATE)
    true_up = S[:, 1] - S[:, 0]
    pre_up = oof_base[0]["pred"][:, 1] - oof_base[0]["pred"][:, 0]
    post_up = ko[0]["pred"][:, 1] - ko[0]["pred"][:, 0]
    corr_pre = float(np.corrcoef(pre_up, true_up)[0, 1])
    corr_post = float(np.corrcoef(post_up, true_up)[0, 1])
    dcorr = corr_post - corr_pre
    print(
        f"  corr 전={corr_pre:+.4f} 후={corr_post:+.4f} Δ={dcorr:+.4f} (문턱 +{KILL_DCORR})"
    )
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v16/spo-plus.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if dcorr < KILL_DCORR:
        print("  킬 스위치 발동 — 의도한 메커니즘으로 게이트 도달 불가, 본 실험 중단")
        out.write_text(
            json.dumps(
                {
                    "candidate": "B spo-plus",
                    "git": git_hash,
                    "killed": True,
                    "kill_reason": "OOF 3.1-uplift corr 개선 미달",
                    "corr_pre": corr_pre,
                    "corr_post": corr_post,
                    "dcorr": dcorr,
                    "threshold": KILL_DCORR,
                    "pass": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"저장: {out}")
        return 0

    print("[3/4] 본 실험 — 15-fold SPO+ (τ=0.01 단독 판정)", flush=True)
    b_spo, oof_spo, diags = spo_candidate(X, Y, S, C, fold_sets, policy, TAU_GATE)
    res_spo = calibrate(
        "spo-plus(τ=0.01)", attach_light_base(b_spo), oof_spo, masks, truth, policy
    )
    delta = res_spo["weighted"] - res_base["weighted"]
    print(f"  spo weighted_cv={res_spo['weighted']:.4f}  Δ={delta:+.4f}")

    print("[4/4] 이중 게이트 + 진단", flush=True)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    spo_cfg = {
        t: (res_spo["tiers"][t]["beta"], res_spo["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(b_base, base_cfg, truth, policy)
    pf_spo, _ = fold_weighted_scores(b_spo, spo_cfg, truth, policy)
    pf_delta = pf_spo - pf_base
    pos = int((pf_delta > 0).sum())
    fx_base, vio_b = fold_weighted_scores(b_base, V13_CONFIG, truth, policy)
    fx_spo, vio_s = fold_weighted_scores(b_spo, V13_CONFIG, truth, policy)
    fixed_delta = float(fx_spo.mean() - fx_base.mean())
    print(
        f"  per-fold Δ>0: {pos}/15 · 고정 캘리브레이션 Δ={fixed_delta:+.4f} (위반 {vio_s})"
    )
    verdict = delta >= GATE_DELTA and fixed_delta >= 0.0 and vio_s == 0
    print(
        f"B 게이트: {'통과' if verdict else '미달 → 기각'} (calibrate +{GATE_DELTA} ∧ 고정 Δ≥0 ∧ 위반 0)"
    )

    # 민감도 (보고 전용, 후보 자격 없음)
    sens = {}
    for tau in TAU_SENS:
        b_s, oof_s, _ = spo_candidate(X, Y, S, C, fold_sets[:1], policy, tau)
        r = calibrate(
            f"spo(τ={tau})", attach_light_base(b_s), oof_s, masks, truth, policy
        )
        sens[str(tau)] = r["weighted"]
        print(f"  [민감도] τ={tau}: seed11 weighted={r['weighted']:.4f}")

    out.write_text(
        json.dumps(
            {
                "candidate": "B spo-plus",
                "git": git_hash,
                "killed": False,
                "corr_pre": corr_pre,
                "corr_post": corr_post,
                "dcorr": dcorr,
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_spo["weighted"],
                "delta": delta,
                "fixed_v13_delta": fixed_delta,
                "fixed_v13_violations": vio_s,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "tau_sensitivity_seed11": sens,
                "fold_diag_sample": diags[0],
                "pass": bool(verdict),
                "tiers": res_spo["tiers"],
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
