# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""A 게이트 — 3.1 uplift 축소 γ family (train 전용, V16-GATES.md 사전 등록).

배경: 예측된 L→M 점수 격차는 참값과 상관 0.033으로 사실상 노이즈다. 노이즈
격차가 그리디 효율 순위에 들어가면 과대예측 문항이 우선 승급되는 역선택이
생긴다. 변환

    ŝ_M' = ŝ_L + γ·(ŝ_M − ŝ_L) + (1−γ)·ḡ_tr,   ḡ_tr = fold-train 참 평균 uplift

으로 격차의 노이즈 성분을 죽인다. γ=1이 항등원(기준선과 비트 동일해야 함).
γ=0이면 3.1 승급 순위가 비용 순으로만 정렬된다 — "넓고 얕은" 전략의 논리적 극한.

등록 arm: 주 γ=0, 헤지 γ=0.25, 참고 C&W 3×3(정직 inner-CV 적합 — 귀속 검증 전용).
이중 게이트: ① calibrate ΔCV ≥ +0.004 ② v1.3 고정 (β,margin)에서 Δ ≥ 0 및
15-fold 예산 위반 0건. 반복 검증(보고 전용): 독립 fold 구조 seed(44,55,66).

누수 자가검사: ḡ_tr·M·c는 fold-train 슬라이스에서만 추정하며, 변환은 예측값의
후처리라 비용 열·smear·σ는 기준선과 비트 동일하다.
"""

from __future__ import annotations

import copy
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
GAMMA_PRIMARY = 0.0
GAMMA_HEDGE = 0.25
INNER_SEED = 7
INNER_FOLDS = 5
BASELINE_CV = 0.6553
GATE_DELTA = 0.004
INTEGRATED_GATE = 0.6613
REP_SEEDS = (44, 55, 66)
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}


def shrink_pred(pred, gbar_tr, gamma):
    """점수 1열(3.1)만 변환. 비용 열 무변경 — 복사본 반환."""
    out = pred.copy()
    out[:, 1] = out[:, 0] + gamma * (pred[:, 1] - pred[:, 0]) + (1.0 - gamma) * gbar_tr
    return out


def transform_candidate(bundles, oof_by_seed, fold_sets, S, gamma):
    """기준선 번들을 후처리해 γ-축소 번들 생성. ḡ_tr는 fold-train 참 라벨만 사용."""
    new_bundles = []
    new_oof = [
        {"pred": o["pred"].copy(), "smear": o["smear"], "sigma": o["sigma"]}
        for o in oof_by_seed
    ]
    bi = 0
    for si, fold_of in enumerate(fold_sets):
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            b = bundles[bi]
            assert np.array_equal(
                b["idx"], np.where(va)[0]
            ), "누수 검사: fold 매핑 불일치"
            gbar_tr = float((S[tr, 1] - S[tr, 0]).mean())
            nb = dict(b)
            nb["pred"] = shrink_pred(b["pred"], gbar_tr, gamma)
            new_bundles.append(nb)
            new_oof[si]["pred"][va] = shrink_pred(
                oof_by_seed[si]["pred"][va], gbar_tr, gamma
            )
            bi += 1
    return new_bundles, new_oof


def cw_candidate(X, Y, keys, bundles, oof_by_seed, fold_sets):
    """참고 arm — 정직 inner-CV로 적합한 C&W 3×3 점수 블록 M (귀속 검증 전용)."""
    new_bundles = []
    new_oof = [
        {"pred": o["pred"].copy(), "smear": o["smear"], "sigma": o["sigma"]}
        for o in oof_by_seed
    ]
    Ms = []
    bi = 0
    for si, fold_of in enumerate(fold_sets):
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            tr_idx = np.where(tr)[0]
            keys_tr = [keys[i] for i in tr_idx]
            inner = group_fold_ids(keys_tr, INNER_SEED, folds=INNER_FOLDS)
            P = np.zeros((len(tr_idx), 3))
            for j in range(INNER_FOLDS):
                iva = inner == j
                itr = ~iva
                Wi = fit_dual_ridge(X[tr][itr], Y[tr][itr], LAM)
                P[iva] = (X[tr][iva] @ Wi)[:, :3]
            A = np.hstack([P, np.ones((P.shape[0], 1))])
            coef, *_ = np.linalg.lstsq(A, Y[tr][:, :3], rcond=None)
            M, c = coef[:3], coef[3]
            Ms.append(M)
            b = bundles[bi]
            nb = dict(b)
            nb["pred"] = b["pred"].copy()
            nb["pred"][:, :3] = b["pred"][:, :3] @ M + c
            new_bundles.append(nb)
            o = oof_by_seed[si]["pred"][va]
            new_oof[si]["pred"][va, :3] = o[:, :3] @ M + c
            bi += 1
    return new_bundles, new_oof, np.mean(Ms, axis=0)


def fold_weighted_scores(bundles, tier_cfgs, truth, policy, capture=False):
    true_score, true_cost = truth["score"], truth["cost"]
    per_fold, violations, choices = [], 0, []
    for b in bundles:
        total = 0.0
        ch = {}
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
            if capture:
                ch[tier] = choice
        per_fold.append(total)
        if capture:
            choices.append(ch)
    return np.array(per_fold), violations, choices


def attach_light_base(bundles, light_costs):
    for b in bundles:
        b["light_base"] = sum(light_costs[i] for i in b["idx"])
    return bundles


def evaluate_arm(
    name, bundles, oof, masks, truth, policy, res_base, pf_base, fx_base, light_costs
):
    res = calibrate(
        name, attach_light_base(bundles, light_costs), oof, masks, truth, policy
    )
    delta = res["weighted"] - res_base["weighted"]
    cfg = {t: (res["tiers"][t]["beta"], res["tiers"][t]["margin"]) for t in TIERS}
    pf, _, _ = fold_weighted_scores(bundles, cfg, truth, policy)
    pf_delta = pf - pf_base
    pos = int((pf_delta > 0).sum())
    fx, vio, _ = fold_weighted_scores(bundles, V13_CONFIG, truth, policy)
    fixed_delta = float(fx.mean() - fx_base.mean())
    gate = delta >= GATE_DELTA and fixed_delta >= 0.0 and vio == 0
    print(
        f"  {name:16s} cv={res['weighted']:.4f} Δ={delta:+.4f} "
        f"고정Δ={fixed_delta:+.4f} 위반={vio} per-fold {pos}/15 → {'통과' if gate else '미달'}"
    )
    return {
        "weighted": res["weighted"],
        "delta": delta,
        "fixed_v13_delta": fixed_delta,
        "fixed_v13_violations": vio,
        "per_fold_positive": pos,
        "per_fold_delta": pf_delta.tolist(),
        "gate": bool(gate),
        "tiers": res["tiers"],
    }


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
    Y = np.zeros((n, 6))
    Y[:, :3] = S
    for i in range(n):
        for j, m in enumerate(MODEL_IDS):
            Y[i, 3 + j] = math.log(max(true_cost[i][m], 1e-9))

    keys = template_group_keys(texts)
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    masks = stress_masks(texts)

    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val

    print("[1/5] 기준선 재현 + γ=1 항등원 검사", flush=True)
    b_base, oof_base = linear_candidate(X, Y, fold_sets, LAM)
    res_base = calibrate(
        "baseline",
        attach_light_base(b_base, light_costs),
        oof_base,
        masks,
        truth,
        policy,
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효")
        return 2
    b_id, _ = transform_candidate(b_base, oof_base, fold_sets, S, 1.0)
    id_diff = max(
        float(np.max(np.abs(a["pred"] - b["pred"]))) for a, b in zip(b_id, b_base)
    )
    print(f"  γ=1 항등원 최대 오차 = {id_diff:.2e}")
    if id_diff > 1e-12:
        print("  선행 관문 실패 — 변환 구현 결함")
        return 3

    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _, _ = fold_weighted_scores(b_base, base_cfg, truth, policy)
    fx_base, _, ch_base = fold_weighted_scores(
        b_base, V13_CONFIG, truth, policy, capture=True
    )

    print("[2/5] 등록 arm 평가 (seed 11/22/33)", flush=True)
    results = {"baseline_weighted": res_base["weighted"]}
    arms = {}
    for label, gamma in (("gamma=0", GAMMA_PRIMARY), ("gamma=0.25", GAMMA_HEDGE)):
        nb, no = transform_candidate(b_base, oof_base, fold_sets, S, gamma)
        arms[label] = (nb, no)
        results[label] = evaluate_arm(
            label, nb, no, masks, truth, policy, res_base, pf_base, fx_base, light_costs
        )
    print("[3/5] 참고 arm — C&W 3×3 (정직 inner-CV, 판정 자격 없음)", flush=True)
    cb, co, M_mean = cw_candidate(X, Y, keys, b_base, oof_base, fold_sets)
    results["cw_3x3_ref"] = evaluate_arm(
        "C&W 3x3 (ref)",
        cb,
        co,
        masks,
        truth,
        policy,
        res_base,
        pf_base,
        fx_base,
        light_costs,
    )
    results["cw_M_mean"] = np.asarray(M_mean).tolist()

    print("[4/5] 반복 검증 — 독립 fold 구조 seed(44,55,66), 보고 전용", flush=True)
    rep_sets = [group_fold_ids(keys, s) for s in REP_SEEDS]
    b_rep, oof_rep = linear_candidate(X, Y, rep_sets, LAM)
    res_rep = calibrate(
        "baseline-rep",
        attach_light_base(b_rep, light_costs),
        oof_rep,
        masks,
        truth,
        policy,
    )
    pf_rep_base, _, _ = fold_weighted_scores(
        b_rep,
        {
            t: (res_rep["tiers"][t]["beta"], res_rep["tiers"][t]["margin"])
            for t in TIERS
        },
        truth,
        policy,
    )
    fx_rep_base, _, _ = fold_weighted_scores(b_rep, V13_CONFIG, truth, policy)
    results["replication"] = {"baseline_weighted": res_rep["weighted"]}
    for label, gamma in (("gamma=0", GAMMA_PRIMARY), ("gamma=0.25", GAMMA_HEDGE)):
        nb, no = transform_candidate(b_rep, oof_rep, rep_sets, S, gamma)
        results["replication"][label] = evaluate_arm(
            f"rep {label}",
            nb,
            no,
            masks,
            truth,
            policy,
            res_rep,
            pf_rep_base,
            fx_rep_base,
            light_costs,
        )

    print("[5/5] 판정 + 배정 이동 진단", flush=True)
    prim = results["gamma=0"]
    hedge = results["gamma=0.25"]
    if prim["gate"] and prim["weighted"] >= INTEGRATED_GATE:
        final = ("gamma=0", prim)
    elif hedge["gate"] and hedge["weighted"] >= INTEGRATED_GATE:
        final = ("gamma=0.25", hedge)
    else:
        final = (None, None)
    # 배정 이동: v1.3 고정 설정에서 주 arm이 바꾼 배정 비율 (tier별)
    _, _, ch_prim = fold_weighted_scores(
        arms["gamma=0"][0], V13_CONFIG, truth, policy, capture=True
    )
    move = {}
    for tier in TIERS:
        tot = chg = 0
        for a, b in zip(ch_base, ch_prim):
            for x, y_ in zip(a[tier], b[tier]):
                tot += 1
                chg += x != y_
        move[tier] = chg / tot
    print("  배정 이동률(v1.3 고정): " + ", ".join(f"{t} {move[t]:.1%}" for t in TIERS))
    if final[0]:
        print(
            f"A 게이트: 통과 — 최종 arm {final[0]} (cv {final[1]['weighted']:.4f}, 통합 게이트 {INTEGRATED_GATE} 충족)"
        )
    else:
        print("A 게이트: 미달 → 기각")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v16/uplift-shrink.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.update(
        {
            "candidate": "A uplift-shrink",
            "git": git_hash,
            "identity_check_max_diff": id_diff,
            "assignment_move_fixed_v13": move,
            "final_arm": final[0],
            "pass": bool(final[0]),
            "leakage_selfcheck": "ḡ_tr·M·c 전부 fold-train 슬라이스에서만 추정 (assert로 fold 매핑 검증)",
        }
    )
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
