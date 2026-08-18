# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""C5 게이트 — 라벨 보존 표기 증강 (train 전용, V15-GATES.md 사전 등록).

같은 문항을 표기만 다르게 쓴 복제본을 학습에만 추가해, 예측기가 서식 차이에
휘둘리지 않게 한다. 변형 두 종은 라벨(모델별 점수·비용)을 바꾸지 않는 표면 정규화다:

  A. 공백 정규화 — 줄 끝 공백 제거, 3연속 이상 빈 줄을 1개로 압축
  B. 문장부호 정규화 — 둥근따옴표→곧은따옴표, em/en대시→하이픈, 숫자 천단위 콤마 제거

원본과 동일해지는 변형은 추가하지 않는다(순수 중복은 가중치 왜곡만 만든다).

fold 누수 방지 설계: 증강 행은 fold 분할에 참여하지 않고 원본의 fold 소속을
그대로 상속한다. 검증·OOF·스트레스·smear·σ·light_base는 전부 원본 행으로만
계산한다 (V15-GATES.md — 복제 잔차의 상관이 σ를 하향시키는 것을 차단).
λ=10 고정(표본 증가에 따른 재격자화 없음, 사전 등록).

게이트: 동일 fold 짝지음 ΔCV ≥ +0.004. 스크리닝 전 기준선 0.6553 재현 필수.
"""

from __future__ import annotations

import json
import math
import re
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

_BLANKLINES = re.compile(r"\n{3,}")
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d{3})")
_PUNCT_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})


def aug_whitespace(text: str) -> str:
    t = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKLINES.sub("\n\n", t)


def aug_punct(text: str) -> str:
    return _DIGIT_COMMA.sub("", text.translate(_PUNCT_MAP))


def build_matrix(texts, extractor=extract_sparse):
    X = np.zeros((len(texts), TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extractor(t).items():
            X[i, idx] = val
    return X


def aug_candidate(X_orig, Y, fold_sets, aug_origin, X_aug, lam):
    """증강 학습 후보 — linear_candidate와 동일 계약의 bundles/oof 반환.

    증강 행은 원본의 fold 소속을 상속해 학습에만 들어간다. 검증 예측·smear·σ는
    원본 행으로만 계산한다.
    """
    aug_origin = np.asarray(aug_origin)
    bundles, oof_by_seed = [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            aug_in_tr = tr[aug_origin]  # 원본이 fold-train인 증강 행만
            X_tr = np.vstack([X_orig[tr], X_aug[aug_in_tr]])
            Y_tr = np.vstack([Y[tr], Y[aug_origin[aug_in_tr]]])
            W = fit_dual_ridge(X_tr, Y_tr, lam)
            pred_tr_orig = X_orig[tr] @ W  # 잔차 통계는 원본 행만
            smear, sigma = cost_stats(pred_tr_orig[:, 3:], Y[tr][:, 3:])
            pred_va = X_orig[va] @ W
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

    def attach_light_base(bundles):
        for b in bundles:
            b["light_base"] = sum(light_costs[i] for i in b["idx"])
        return bundles

    # 증강 표본 구성 — 원본과 달라지는 변형만
    aug_origin, aug_texts = [], []
    n_ws = n_punct = 0
    for i, t in enumerate(texts):
        for fn in (aug_whitespace, aug_punct):
            v = fn(t)
            if v != t:
                aug_origin.append(i)
                aug_texts.append(v)
                if fn is aug_whitespace:
                    n_ws += 1
                else:
                    n_punct += 1
    print(f"증강 표본: {len(aug_texts)}개 (공백 {n_ws}, 문장부호 {n_punct}) / 원본 {n}")

    print("[1/3] 기준선 재현 (증강 없음, λ=10)", flush=True)
    X_orig = build_matrix(texts)
    b_base, oof_base = linear_candidate(X_orig, Y, fold_sets, LAM)
    res_base = calibrate(
        "baseline", attach_light_base(b_base), oof_base, masks, truth, policy
    )
    print(
        f"  baseline weighted_cv={res_base['weighted']:.4f} (등록 기준선 {BASELINE_CV})"
    )
    if abs(res_base["weighted"] - BASELINE_CV) > 0.0005:
        print("  경고: 기준선 재현 실패 — 결과 무효, 프로토콜 점검 필요")
        return 2

    print("[2/3] 변형 평가 (증강 학습, λ=10)", flush=True)
    X_aug = build_matrix(aug_texts)
    b_aug, oof_aug = aug_candidate(X_orig, Y, fold_sets, aug_origin, X_aug, LAM)
    res_aug = calibrate(
        "linear+augment", attach_light_base(b_aug), oof_aug, masks, truth, policy
    )
    delta = res_aug["weighted"] - res_base["weighted"]
    print(f"  augment weighted_cv={res_aug['weighted']:.4f}  Δ={delta:+.4f}")
    for tier, cfg in res_aug["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )

    print("[3/3] 진단 — paired per-fold Δ + v1.3 고정 캘리브레이션", flush=True)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    aug_cfg = {
        t: (res_aug["tiers"][t]["beta"], res_aug["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(
        attach_light_base(b_base), base_cfg, truth, policy
    )
    pf_aug, _ = fold_weighted_scores(attach_light_base(b_aug), aug_cfg, truth, policy)
    pf_delta = pf_aug - pf_base
    pos = int((pf_delta > 0).sum())
    print(f"  per-fold Δ>0: {pos}/15 (참고 지표 ≥10)")

    fx_base, vio_b = fold_weighted_scores(
        attach_light_base(b_base), V13_CONFIG, truth, policy
    )
    fx_aug, vio_a = fold_weighted_scores(
        attach_light_base(b_aug), V13_CONFIG, truth, policy
    )
    fixed_delta = float(fx_aug.mean() - fx_base.mean())
    print(
        f"  v1.3 고정 캘리브레이션 Δ={fixed_delta:+.4f} (예산 위반 base {vio_b}, aug {vio_a})"
    )

    verdict = delta >= GATE_DELTA
    print(f"C5 게이트: {'통과' if verdict else '미달 → 기각'} (문턱 +{GATE_DELTA})")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v15/augment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate": "C5 augment",
                "git": git_hash,
                "n_augmented": len(aug_texts),
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_aug["weighted"],
                "delta": delta,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "fixed_v13_delta": fixed_delta,
                "pass": bool(verdict),
                "tiers": res_aug["tiers"],
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
