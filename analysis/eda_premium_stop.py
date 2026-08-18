# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""Premium allocate 종료 원인 진단 (train 전용 · 기록 전용 — 게이트 아님).

질문: v1.6 Premium이 Dev에서 예산의 60.7%(2.428/4.0)만 쓰고 남기는 이유가
예산 소진(β 비관이 만든 자기 기준 소진)인지 후보 소진(K1 증분 미생성)인지.

방법: os2_policy.allocate와 동일 로직의 계측 fork로 문항별 증분 생성 경로와
증분별 기각 사유를 센다. allocate 루프에는 break가 없으므로 "멈춘 시점"이
아니라 증분 단위 사유(채택 / 예산 skip / 선행 스텝 연쇄 skip)로 재야 한다.
연쇄 skip은 같은 문항의 선행 스텝이 예산 skip을 당했을 때만 생기므로
사실상 예산 사유의 하위 분류다.

선행 관문: 계측 fork는 모든 실행에서 원본 allocate와 완전히 같은 choice를
반환해야 한다 (선례: exp_hetero_sigma.py의 fork 동등성 관문). 불일치 시 무효.

두 모드:
  1) deployed — 전체 train × 배포 아티팩트(learned-router.v1.json), 제출 구성 그대로.
     ŝ_M − ŝ_L 상수(+0.08125) 검증과 비관 쐐기 상수도 여기서 실측한다.
  2) cv — rematch 15-fold(seed 11/22/33)에서 v1.6 예측기(γ=0 축소)와
     v1.3 예측기(원 헤드)를 각자의 Premium (β, margin)으로 대조.

dev 미사용 · 선택 개입 없음 · 채택 대상 없음.
산출: build/v16/premium-stop-diag.json
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np

from exp_uplift_shrink import V13_CONFIG, attach_light_base, transform_candidate
from os2_features import TOTAL_DIM, extract_sparse
from os2_policy import allocate
from rematch import (
    CV_SEEDS,
    episode_text,
    group_fold_ids,
    linear_candidate,
    realized,
    rows_with_pessimism,
    template_group_keys,
)
from ossp_router.learned_router import load_bundled_artifact, predict_episode
from ossp_router.protocol import (
    MODEL_IDS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

LAM = 10.0
TIER = "premium"
V16_PREMIUM = (1.0, 0.92)
V13_PREMIUM = V13_CONFIG["premium"]  # (1.0, 0.80)
ARTIFACT_PATH = REPO / "src/ossp_router/resources/learned-router.v1.json"


def allocate_traced(predictions, budget_multiplier, margin):
    """os2_policy.allocate의 계측 사본 — choice는 원본과 동일해야 한다(관문).

    반환 diag:
      path[i]      — 증분 생성 경로: two_step / k1_merged / m_only / k1_only / none
      outcome[(i, step)] — (사유, target): accepted / budget_skip / chain_skip
      est_base, budget, spent — 비관 추정 예산 축
    """
    n = len(predictions)
    choice = ["ax31-light"] * n
    est_base = sum(p["ax31-light"][1] for p in predictions)
    diag = {"est_base": est_base, "path": ["none"] * n, "outcome": {}}
    if est_base <= 0:
        diag.update(budget=0.0, spent=0.0)
        return choice, diag
    budget = est_base * budget_multiplier * margin
    spent = est_base

    increments = []
    for i, p in enumerate(predictions):
        s_l, c_l = p["ax31-light"]
        s_a, c_a = p["ax31"]
        s_k, c_k = p["axk1-think"]
        c_l = max(c_l, 1e-12)
        c_a = max(c_a, c_l * 1.01)
        c_k = max(c_k, c_a * 1.01)
        steps = []
        if s_a > s_l and s_k > s_a:
            r1 = (s_a - s_l) / (c_a - c_l)
            r2 = (s_k - s_a) / (c_k - c_a)
            if r2 >= r1:
                steps = [("axk1-think", s_k - s_l, c_k - c_l)]
                diag["path"][i] = "k1_merged"
            else:
                steps = [
                    ("ax31", s_a - s_l, c_a - c_l),
                    ("axk1-think", s_k - s_a, c_k - c_a),
                ]
                diag["path"][i] = "two_step"
        elif s_a > s_l:
            steps = [("ax31", s_a - s_l, c_a - c_l)]
            diag["path"][i] = "m_only"
        elif s_k > s_l:
            steps = [("axk1-think", s_k - s_l, c_k - c_l)]
            diag["path"][i] = "k1_only"
        tie = (
            p["ax31-light"][0],
            p["ax31-light"][1],
            p["ax31"][0],
            p["ax31"][1],
            p["axk1-think"][0],
            p["axk1-think"][1],
        )
        for step_index, (target, ds, dc) in enumerate(steps):
            increments.append((ds / dc, tie, step_index, i, target, ds, dc))

    increments.sort(key=lambda item: (-item[0], item[1], item[2]))
    taken_step = [-1] * n
    for ratio, tie, step_index, i, target, ds, dc in increments:
        if step_index != taken_step[i] + 1:
            diag["outcome"][(i, step_index)] = ("chain_skip", target)
            continue
        if spent + dc > budget:
            diag["outcome"][(i, step_index)] = ("budget_skip", target)
            continue
        spent += dc
        choice[i] = target
        taken_step[i] = step_index
        diag["outcome"][(i, step_index)] = ("accepted", target)
    diag.update(budget=budget, spent=spent)
    return choice, diag


def run_traced(rows, mult, margin):
    """원본 allocate와 대조하며 계측 실행. 불일치 수를 함께 반환."""
    reference = allocate(rows, mult, margin)
    choice, diag = allocate_traced(rows, mult, margin)
    mismatch = sum(a != b for a, b in zip(reference, choice))
    return choice, diag, mismatch


def summarize(choice, diag):
    """문항별 K1 결말과 증분 기각 사유를 집계한다."""
    n = len(choice)
    per_target = {"ax31": Counter(), "axk1-think": Counter()}
    k1_status = {}
    for (i, step_index), (status, target) in diag["outcome"].items():
        per_target[target][status] += 1
        if target == "axk1-think":
            k1_status[i] = status
    k1 = Counter()
    for i in range(n):
        if i not in k1_status:
            k1["absent_structural"] += 1  # 증분 자체가 안 생김 (m_only / none)
        elif choice[i] == "axk1-think":
            k1["taken"] += 1
        elif k1_status[i] == "budget_skip":
            k1["budget_direct"] += 1
        else:
            k1["budget_chain"] += 1  # 선행 3.1 스텝이 예산 skip → 연쇄 배제
    budget = diag["budget"]
    est_base = diag["est_base"]
    return {
        "paths": dict(Counter(diag["path"])),
        "increment_outcomes": {t: dict(c) for t, c in per_target.items()},
        "k1": dict(k1),
        "k1_absent_rate": k1["absent_structural"] / n,
        "pessimistic": {
            "est_base": est_base,
            "budget": budget,
            "spent": diag["spent"],
            "budget_utilization": diag["spent"] / budget if budget else 0.0,
            "predicted_ratio_vs_light": (diag["spent"] / est_base if est_base else 0.0),
        },
        "choice_counts": dict(Counter(choice)),
    }


def main() -> int:
    policy = load_bundled_policy()
    artifact = load_bundled_artifact()
    if artifact is None:
        print("오류: 배포 아티팩트를 찾을 수 없습니다.", file=sys.stderr)
        return 2
    mult = float(policy.tiers[TIER].budget_multiplier)
    if artifact.tier_config[TIER] != V16_PREMIUM:
        print("오류: 아티팩트 Premium (β, margin)이 v1.6 등록값과 다릅니다.")
        return 2
    beta, margin = V16_PREMIUM

    inputs = load_input(REPO / "data/materialized/train/inputs.json")
    outcomes = load_outcomes(REPO / "data/train/outcomes.json")
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    rates = {m: policy.models[m] for m in MODEL_IDS}
    score_map, token_map = {}, {}
    for oc in outcomes.outcomes:
        score_map[(oc.episode_id, oc.model_id)] = float(oc.score)
        token_map[(oc.episode_id, oc.model_id)] = (oc.input_tokens, oc.output_tokens)
    eids = [ep.episode_id for ep in inputs.episodes]

    def true_cost_of(eid, m):
        ti, to = token_map[(eid, m)]
        r = rates[m]
        return (
            float(r.fixed_cost)
            + (ti * float(r.input_token_rate) + to * float(r.output_token_rate))
            / policy.token_unit
        )

    true_score = [{m: score_map[(e, m)] for m in MODEL_IDS} for e in eids]
    true_cost = [{m: true_cost_of(e, m) for m in MODEL_IDS} for e in eids]
    light_costs = [c[MODEL_IDS[0]] for c in true_cost]
    truth_score, truth_cost = true_score, true_cost

    total_mismatch = 0
    results = {
        "diagnostic": "premium-stop",
        "question": "Premium 잔여 예산의 원인 — 예산 소진(β 비관) vs 후보 소진(K1 증분 미생성)",
        "protocol": {
            "dev_used": False,
            "selection_intervention": False,
            "gate": "없음 — 기록 전용 회고 진단, 채택 대상 없음",
        },
    }

    print("[1/3] deployed — 전체 train × 배포 아티팩트 (제출 구성 그대로)", flush=True)
    pess = {
        m: 1.0 if m == MODEL_IDS[0] else math.exp(beta * artifact.cost_sigma[m])
        for m in MODEL_IDS
    }
    rows_deployed, gaps = [], []
    for ep in inputs.episodes:
        base = predict_episode(ep, artifact)
        gaps.append(base[MODEL_IDS[1]][0] - base[MODEL_IDS[0]][0])
        rows_deployed.append({m: (s, c * pess[m]) for m, (s, c) in base.items()})
    choice_d, diag_d, mismatch = run_traced(rows_deployed, mult, margin)
    total_mismatch += mismatch
    summ_d = summarize(choice_d, diag_d)
    idx_all = list(range(n))
    light_base = sum(light_costs)
    s_train, u_train = realized(idx_all, choice_d, truth_score, truth_cost, light_base)
    gaps = np.array(gaps)
    sigma_k1 = artifact.cost_sigma[MODEL_IDS[2]]
    summ_d["realized_train"] = {"score": s_train, "used_ratio": u_train}
    summ_d["m_head_gap"] = {
        "min": float(gaps.min()),
        "max": float(gaps.max()),
        "n_off_constant": int((np.abs(gaps - 0.08125) > 1e-6).sum()),
    }
    summ_d["wedge"] = {
        "sigma_k1": sigma_k1,
        "exp_beta_sigma_k1": math.exp(beta * sigma_k1),
        "smear_k1": artifact.cost_smear[MODEL_IDS[2]],
    }
    results["deployed"] = {"config": {"beta": beta, "margin": margin, "mult": mult}}
    results["deployed"].update(summ_d)
    p = summ_d["pessimistic"]
    print(
        f"  경로: {summ_d['paths']} | K1 결말: {summ_d['k1']}\n"
        f"  비관 예산 소진율 {p['budget_utilization']:.4%}, "
        f"train 실현 사용률 {u_train:.4f}/{mult} ({u_train / mult:.1%}), "
        f"fork 불일치 {mismatch}"
    )

    print("[2/3] cv — 15-fold, v1.6(γ=0)·v1.3(원 헤드) 대조", flush=True)
    S = np.array([[truth_score[i][m] for m in MODEL_IDS] for i in range(n)])
    Y = np.zeros((n, 6))
    Y[:, :3] = S
    for i in range(n):
        for j, m in enumerate(MODEL_IDS):
            Y[i, 3 + j] = math.log(max(truth_cost[i][m], 1e-9))
    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val
    keys = template_group_keys(texts)
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    b_base, oof_base = linear_candidate(X, Y, fold_sets, LAM)
    b_v16, _ = transform_candidate(b_base, oof_base, fold_sets, S, 0.0)
    truth = {"score": truth_score, "cost": truth_cost, "light_costs": light_costs}

    for label, bundles, (cv_beta, cv_margin) in (
        ("cv_v16", b_v16, V16_PREMIUM),
        ("cv_v13_contrast", b_base, V13_PREMIUM),
    ):
        attach_light_base(bundles, light_costs)
        per_fold = []
        for b in bundles:
            rows = rows_with_pessimism(b["pred"], b["smear"], b["sigma"], cv_beta)
            ch, dg, mm = run_traced(rows, mult, cv_margin)
            total_mismatch += mm
            summ = summarize(ch, dg)
            s, u = realized(b["idx"], ch, truth_score, truth_cost, b["light_base"])
            per_fold.append(
                {
                    "k1_absent_rate": summ["k1_absent_rate"],
                    "k1": summ["k1"],
                    "pessim_budget_utilization": summ["pessimistic"][
                        "budget_utilization"
                    ],
                    "realized_used": u,
                    "realized_score": s,
                }
            )
        arr = lambda key: np.array([f[key] for f in per_fold])
        results[label] = {
            "config": {"beta": cv_beta, "margin": cv_margin, "mult": mult},
            "per_fold": per_fold,
            "summary": {
                key: {
                    "min": float(arr(key).min()),
                    "median": float(np.median(arr(key))),
                    "max": float(arr(key).max()),
                }
                for key in (
                    "k1_absent_rate",
                    "pessim_budget_utilization",
                    "realized_used",
                )
            },
        }
        sm = results[label]["summary"]
        print(
            f"  {label}: K1 미생성률 med {sm['k1_absent_rate']['median']:.1%} "
            f"[{sm['k1_absent_rate']['min']:.1%}, {sm['k1_absent_rate']['max']:.1%}], "
            f"비관 소진율 med {sm['pessim_budget_utilization']['median']:.2%}, "
            f"실현 사용 med {sm['realized_used']['median']:.3f}/{mult}"
        )

    print("[3/3] 관문·기록", flush=True)
    results["fork_equivalence_mismatch"] = total_mismatch
    if total_mismatch:
        print(f"  선행 관문 실패 — fork 불일치 {total_mismatch}건, 결과 무효")
        return 3
    print("  fork 동등성: 전 실행 choice 완전 일치 (불일치 0)")

    results["git"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    results["artifact_sha256"] = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    out = REPO / "build/v16/premium-stop-diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
