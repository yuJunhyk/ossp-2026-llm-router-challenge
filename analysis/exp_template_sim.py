# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""C1 게이트 — 템플릿 유사도 특징 (train 전용, V15-GATES.md 사전 등록).

train의 템플릿 계열(정규화 25자 접두사 공유)이 756문항이라는 사실을 특징으로
쓴다. 문항마다 "은행(bank)과 얼마나 닮은 접두사인가"를 수치 3개로 재서
수치 여유 슬롯(인덱스 34·35·36)에 넣는다. TOTAL_DIM 불변.

  f34 = 은행에 같은 25자 접두사 문항이 존재하는가 (이진)
  f35 = 은행과의 최장 공통 접두사 길이 / 25 (캡 [0,1] — 외삽 방지)
  f36 = log1p(min(같은 접두사 문항 수, 20))

fold 누수 방지 설계:
  - CV에서 은행은 fold-train 접두사만으로 구성한다.
  - fold-train 행 자신의 특징은 자기 자신을 은행에서 제외하고 계산한다
    (포함하면 모든 학습 행에서 f34=1로 퇴화).
  - 그룹 fold 특성상 검증 문항의 계열 전체가 검증 쪽에 있으므로, CV의 Δ는
    배포(전체 train 은행) 이득의 하한 추정이다 — V15-GATES.md에 등록됨.

게이트: 동일 fold 짝지음 ΔCV ≥ +0.004. 스크리닝 전 기준선 0.6553 재현 필수.
채택 시 배포 배관(아티팩트에 은행 동봉 + parity 재설계)은 별도 단계.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np

from rematch import (
    CV_SEEDS,
    FOLDS,
    GROUP_PREFIX,
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
BASELINE_CV = 0.6553
GATE_DELTA = 0.004
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}
SLOT = 34  # 수치 여유 슬롯 시작 (bias=33 뒤, NUMERIC_DIM=40 미만)
COUNT_CAP = 20


def norm_prefix(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:GROUP_PREFIX]


def _lcp(a: str, b: str) -> int:
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m


class PrefixBank:
    """정렬 접두사 은행 — 존재/최장 공통 접두사/동일 접두사 수를 O(log n)에 답한다."""

    def __init__(self, prefixes: list[str]):
        self.counter = Counter(prefixes)
        self.sorted_unique = sorted(self.counter)

    def features(self, prefix: str, exclude_self: bool) -> tuple[float, float, float]:
        count = self.counter.get(prefix, 0) - (1 if exclude_self else 0)
        if count > 0:
            return 1.0, 1.0, math.log1p(min(count, COUNT_CAP))
        # 정확 일치 없음 — 이웃 접두사와의 LCP
        best = 0
        pos = bisect.bisect_left(self.sorted_unique, prefix)
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < len(self.sorted_unique):
                cand = self.sorted_unique[j]
                if cand == prefix:
                    continue  # 자기 자신뿐인 항목 (exclude_self로 count 0)
                best = max(best, _lcp(prefix, cand))
        return 0.0, best / GROUP_PREFIX, 0.0


def build_matrix(texts):
    X = np.zeros((len(texts), TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val
    return X


def sim_candidate(X_base, Y, fold_sets, prefixes, lam):
    """fold-train 은행으로 유사도 특징을 만든 뒤 적합 — linear_candidate 계약 준수."""
    n = X_base.shape[0]
    bundles, oof_by_seed = [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            bank = PrefixBank([prefixes[i] for i in np.where(tr)[0]])
            X_var = X_base.copy()
            for i in range(n):
                f1, f2, f3 = bank.features(prefixes[i], exclude_self=bool(tr[i]))
                X_var[i, SLOT] = f1
                X_var[i, SLOT + 1] = f2
                X_var[i, SLOT + 2] = f3
            W = fit_dual_ridge(X_var[tr], Y[tr], lam)
            pred_tr = X_var[tr] @ W
            smear, sigma = cost_stats(pred_tr[:, 3:], Y[tr][:, 3:])
            pred_va = X_var[va] @ W
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

    full_bank = PrefixBank(prefixes)
    in_family = sum(
        1 for p in prefixes if full_bank.features(p, exclude_self=True)[0] > 0
    )
    print(f"전체 train 기준 계열 소속(자기 제외 후 동일 접두사 존재): {in_family}/{n}")

    print("[1/3] 기준선 재현 (extract_sparse, λ=10)", flush=True)
    X_base = build_matrix(texts)
    b_base, oof_base = linear_candidate(X_base, Y, fold_sets, LAM)
    res_base = calibrate(
        "baseline", attach_light_base(b_base), oof_base, masks, truth, policy
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효, 프로토콜 점검 필요")
        return 2

    print("[2/3] 변형 평가 (fold-train 은행 유사도 특징, λ=10)", flush=True)
    b_sim, oof_sim = sim_candidate(X_base, Y, fold_sets, prefixes, LAM)
    res_sim = calibrate(
        "linear+template-sim", attach_light_base(b_sim), oof_sim, masks, truth, policy
    )
    delta = res_sim["weighted"] - res_base["weighted"]
    print(f"  template-sim weighted_cv={res_sim['weighted']:.4f}  Δ={delta:+.4f}")
    for tier, cfg in res_sim["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )

    print("[3/3] 진단 — paired per-fold Δ + v1.3 고정 캘리브레이션", flush=True)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    sim_cfg = {
        t: (res_sim["tiers"][t]["beta"], res_sim["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(
        attach_light_base(b_base), base_cfg, truth, policy
    )
    pf_sim, _ = fold_weighted_scores(attach_light_base(b_sim), sim_cfg, truth, policy)
    pf_delta = pf_sim - pf_base
    pos = int((pf_delta > 0).sum())
    print(f"  per-fold Δ>0: {pos}/15 (참고 지표 ≥10)")

    fx_base, vio_b = fold_weighted_scores(
        attach_light_base(b_base), V13_CONFIG, truth, policy
    )
    fx_sim, vio_s = fold_weighted_scores(
        attach_light_base(b_sim), V13_CONFIG, truth, policy
    )
    fixed_delta = float(fx_sim.mean() - fx_base.mean())
    print(
        f"  v1.3 고정 캘리브레이션 Δ={fixed_delta:+.4f} (예산 위반 base {vio_b}, sim {vio_s})"
    )

    verdict = delta >= GATE_DELTA
    print(f"C1 게이트: {'통과' if verdict else '미달 → 기각'} (문턱 +{GATE_DELTA})")
    print("주: CV Δ는 배포(전체 은행) 이득의 하한 추정 (V15-GATES.md)")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v15/template-sim.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate": "C1 template-sim",
                "git": git_hash,
                "family_episodes_full_bank": in_family,
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_sim["weighted"],
                "delta": delta,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "fixed_v13_delta": fixed_delta,
                "pass": bool(verdict),
                "tiers": res_sim["tiers"],
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
