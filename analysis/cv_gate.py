#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""train 전용 교차검증 게이트 — 캘리브레이션(margin(n))과 가드 후보를 Dev 없이 판정한다.

절차 (docs/decisions.md ③의 규율을 코드로 고정):
1. 템플릿 그룹 5-fold × 3 seed. DeepMind Mathematics는 생성 모듈(템플릿)별 그룹을 통째로 한
   fold에 넣고, 나머지 출처는 출처 가족 안에서 층화한다. 기본 --grouping template은 AIME도
   층화한다(학습에 AIME가 있는 실제 조건). --grouping source-holdout은 AIME 연도 전체를 한 fold에
   넣어 "처음 보는 출처" 시나리오를 만든다 — 보고용 스트레스이며 선택에는 template을 쓴다.
2. fold 밖(out-of-fold) 예측만 쓴다. fold마다 train_linear와 같은 절차로 다시 적합한다
   (KFEAT z-점수 접기, dual ridge λ=10, ax31 점수 헤드 상수 대체, smear/σ는 학습 fold 잔차).
   입력 split의 selection 파일(그룹 라벨)만 읽고 다른 split의 파일은 열지 않는다.
3. 후보마다 다음을 잰다.
   - fold 게이트: 15개 fold 전부 실현 비용비 ≤ 한도 × {0.90, 0.95, 0.95}
   - 부분집합 게이트: OOF 풀(1,760)에서 비복원으로 뽑은 n ∈ {100, 200, 400, 800, 880} 배치
     200개의 예산 통과율 — n≥800은 100%, 400은 ≥99%, 200은 ≥97%, 100은 ≥95%
   - 출처 편향 게이트: 출처 가족 단독 배치 전부 ≤ 한도 × 0.98, 반가족(50%) 배치 n=400 통과율 ≥ 98%
   - 배포 규모 게이트: n=880 배치 200개의 최악 비용비 ≤ 한도 × 0.98. fold 크기(≈352)가
     deep_min_episodes보다 작아 깊은 margin이 fold 게이트·vpCV에 전혀 반영되지 않는다는
     검토 지적으로 추가한 게이트다 (R1.5 검토 후 추가 — 사전 등록분이 아님).
   - 배포 규모 출처 편향 게이트: 출처 가족 단독 배치를 깊은 margin(n ≥ deep_min_episodes일 때
     쓰는 값)으로 강제 배분해 전부 ≤ 한도 × 0.98. 비공개 평가셋이 800문항 이상이면서 한 출처에
     치우친 경우를 본뜬다 — 가족 단독 배치는 크기가 작아 그냥 두면 얕은 margin만 검사된다
     (역시 R1.5 검토 후 추가).
   - 목적함수 vpCV: fold별 등급 점수에 예산 초과 fold를 0점 처리한 15-fold 평균
4. 게이트를 전부 통과한 후보 중 vpCV 최대(동률이면 n=880 평균 점수가 높은 쪽)를 고른다.
   등급별 캘리브레이션은 독립이므로 등급별로 고르고, 가드 변형은 세 등급 가중합으로 비교한다.
   어떤 게이트를 현행 구성을 포함한 후보 전원이 못 넘으면(이 OOF 절차가 팀의 원래 절차보다
   가혹해 생기는 일) 그 게이트만 완충을 1.0(실제 한도)으로 완화해 다시 고르고, 완화한 게이트를
   보고서 selection.amended_gates에 남긴다. 부분집합 게이트는 완화하지 않는다.

실행: PYTHONPATH=src python3 analysis/cv_gate.py --report build/cv-gate-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import random
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402

import os2_features  # noqa: E402
from os2_features import (  # noqa: E402
    KFEAT_DIM,
    KFEAT_SLOT,
    TOTAL_DIM,
    _AIME_RE,
    _CODE_EXEC_RE,
    _MC_HEAD_RE,
    _RULE_CHAIN_RE,
    extract_sparse,
    k_guard,
)
from os2_policy import allocate  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from train_linear import (  # noqa: E402
    BIAS_SLOT,
    GAMMA,
    LAMBDA,
    episode_text,
    fit_dual_ridge,
)

SEEDS = (0, 1, 2)
K_FOLDS = 5
FOLD_GATE = {"fast": 0.90, "balanced": 0.95, "premium": 0.95}
SUBSAMPLE_SIZES = (100, 200, 400, 800, 880)
SUBSAMPLE_REPS = 200
SUBSAMPLE_MIN_PASS = {100: 0.95, 200: 0.97, 400: 0.99, 800: 1.0, 880: 1.0}
FAMILY_GATE = 0.98
DEPLOY_GATE = 0.98  # n=880 배치 최악 비용비 ≤ 한도 × DEPLOY_GATE
HALF_FAMILY_N = 400
HALF_FAMILY_REPS = 30
HALF_FAMILY_MIN_PASS = 0.98
WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

# ---------------------------------------------------------------- 가드 후보
# 각 항목: (이름, 모드, 술어). 모드 K = think 점수를 ax31 점수로, MK = ax31·think 점수를 light 점수로.
_HANGUL = re.compile(r"[가-힣]")
_POLY_POW = re.compile(r"\*\*\s*[2-9]|\^\s*[2-9]")
_EQ_ZERO = re.compile(r"=\s*0\s*[.?]?\s*$", re.M)
_BIG7 = re.compile(r"\d{7,}")
_BIG5 = re.compile(r"\d{5,}")
_KGUARD_WIDE_PAT = re.compile(r"\b(prime|composite|prime factors?|factors? of)\b", re.I)
_LATEX_DOLLAR = re.compile(r"\$")


def guard_current(t: str) -> bool:
    return k_guard(t)


def guard_code_exec(t: str) -> bool:
    return bool(_CODE_EXEC_RE.search(t[:6000]))


def guard_poly_big(t: str) -> bool:
    h = t[:6000]
    return bool(_POLY_POW.search(h) and _EQ_ZERO.search(h) and _BIG7.search(h))


def guard_prime_wide(t: str) -> bool:
    h = t[:6000]
    return bool(_KGUARD_WIDE_PAT.search(h) and _BIG5.search(h))


_PRIME_ONLY_PAT = re.compile(r"\b(prime|composite)\b", re.I)


def guard_drop_factors_of(t: str) -> bool:
    """현행 가드에서 `factors? of` 어휘만 뺀 변형 — 'greatest common factor of …'(train-0594, K가 싸고 정답)를 풀어준다."""
    h = t[:6000]
    if not _BIG7.search(h):
        return False
    return bool(_PRIME_ONLY_PAT.search(h) or os2_features._KGUARD_POLY_EQ0.search(h))


def guard_aime_latex(t: str) -> bool:
    h = t[:6000]
    return len(_LATEX_DOLLAR.findall(h)) >= 4 and bool(_AIME_RE.search(h))


GUARD_VARIANTS_ALL = {
    # 이름: [(술어, 모드), ...] — 현행 k_guard(K)는 항상 포함
    "G0-current": [(guard_current, "K")],
    "G1-code-exec-K": [(guard_current, "K"), (guard_code_exec, "K")],
    "G2-poly-big-K": [(guard_current, "K"), (guard_poly_big, "K")],
    "G3-poly-big-MK": [(guard_current, "K"), (guard_poly_big, "MK")],
    "G4-prime-wide-K": [(guard_prime_wide, "K")],
    "G5-aime-latex-K": [(guard_current, "K"), (guard_aime_latex, "K")],
    "G6-poly-MK+prime-wide-K": [(guard_prime_wide, "K"), (guard_poly_big, "MK")],
    "G8-drop-factors-of-K": [(guard_drop_factors_of, "K")],
}

GUARD_VARIANTS = GUARD_VARIANTS_ALL  # --guards 로 부분집합 선택


def guard_definitions() -> dict:
    """보고서에 남기는 가드 정의 — 'G0-current'가 어느 k_guard였는지 파일만 보고 알 수 있게."""
    src = inspect.getsource(os2_features.k_guard)
    patterns = {
        name: getattr(os2_features, name).pattern
        for name in dir(os2_features)
        if name.startswith("_KGUARD_")
    }
    return {
        "current_k_guard_source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "current_k_guard_patterns": patterns,
        "candidates": {
            "guard_code_exec": _CODE_EXEC_RE.pattern,
            "guard_poly_big": f"{_POLY_POW.pattern} AND {_EQ_ZERO.pattern} AND {_BIG7.pattern}",
            "guard_prime_wide": f"{_KGUARD_WIDE_PAT.pattern} AND {_BIG5.pattern}",
            "guard_aime_latex": f"count({_LATEX_DOLLAR.pattern})>=4 AND {_AIME_RE.pattern}",
        },
    }


# ---------------------------------------------------------------- 캘리브레이션 후보
# margin(n) = shallow (n < n0) / deep (n >= n0). n0=None이면 상수 margin.
FAST_GRID = [(m, None, None) for m in (0.94, 0.96, 0.97, 0.98, 1.00, 1.03, 1.06)] + [
    (0.94, 1.03, 400), (0.94, 1.00, 800), (0.97, 1.03, 800), (0.94, 1.06, 800)
]
BALANCED_GRID = [(1.08, None, None), (1.04, None, None), (1.00, None, None)] + [
    (s, d, n0) for s in (0.90, 0.94, 0.96, 1.00) for d in (1.00, 1.04, 1.08) for n0 in (400, 800)
]
PREMIUM_GRID = [
    (0.96, 1.08, 800), (0.96, None, None), (0.96, 1.04, 800), (0.96, 1.00, 800),
    (0.90, 1.08, 800), (0.90, 1.04, 800), (1.00, 1.08, 800),
]
CURRENT = {"fast": (0.94, None, None), "balanced": (1.08, None, None), "premium": (0.96, 1.08, 800)}


def margin_at(cfg, n: int) -> float:
    shallow, deep, n0 = cfg
    if deep is not None and n0 is not None and n >= n0:
        return deep
    return shallow


def cfg_name(cfg) -> str:
    shallow, deep, n0 = cfg
    return f"{shallow:.2f}" if deep is None else f"{shallow:.2f}/{deep:.2f}@{n0}"


# ---------------------------------------------------------------- 자료·그룹
def family_of(eid: str, text: str, dm_module: dict, aime_year: dict) -> tuple[str, str | None]:
    """(가족, 그룹) — 그룹이 None이면 가족 안에서 층화, 아니면 그룹 통째로 한 fold."""
    if eid in dm_module:
        return "deepmind-math", "dm:" + dm_module[eid]
    if eid in aime_year:
        return "aime", f"aime:{aime_year[eid]}"
    head = text[:6000]
    if _CODE_EXEC_RE.search(head):
        return "cruxeval", None
    if len(_HANGUL.findall(head)) / max(1, len(head)) > 0.15:
        return ("belebele-ko" if re.search(r"(?:^|\n)[A-D]\.\s", head) else "hrmcr-ko"), None
    if len(text) > 8000:
        return "babilong", None
    if _RULE_CHAIN_RE.search(head):
        return "ruletaker", None
    if _MC_HEAD_RE.search(head) or re.search(r"(?:^|\n)[A-E]\.\s", head):
        return "truthfulqa", None
    return "gsm8k-other", None


def load_all(inputs_path: Path, outcomes_path: Path, grouping: str = "template"):
    """grouping: 'template'는 DM 모듈만 그룹(AIME는 층화 — 학습에 AIME가 있는 실제 조건),
    'source-holdout'는 AIME 연도도 통째로 fold에 넣어 미학습 출처 시나리오를 만든다."""
    policy = load_bundled_policy()
    inputs = load_input(inputs_path)
    outcomes = load_outcomes(outcomes_path)
    ids = [ep.episode_id for ep in inputs.episodes]
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(ids)
    rates = {m: policy.models[m] for m in MODEL_IDS}
    truth_score = np.zeros((n, 3))
    truth_cost = np.zeros((n, 3))
    index = {eid: i for i, eid in enumerate(ids)}
    for oc in outcomes.outcomes:
        r = rates[oc.model_id]
        j = MODEL_IDS.index(oc.model_id)
        cost = float(r.fixed_cost) + (
            oc.input_tokens * float(r.input_token_rate) + oc.output_tokens * float(r.output_token_rate)
        ) / policy.token_unit
        truth_score[index[oc.episode_id], j] = float(oc.score)
        truth_cost[index[oc.episode_id], j] = cost
    X = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X[i, idx] = val
    Y = np.zeros((n, 6))
    Y[:, :3] = truth_score
    Y[:, 3:] = np.log(np.maximum(truth_cost, 1e-9))
    split = inputs.split
    dm = json.loads((REPO / "data/sources/deepmind-mathematics-selection.v1.json").read_text())
    dm_module = {
        e["episode_id"]: e["sample_id"].split(":")[1] for e in dm["splits"].get(split, [])
    }
    aime_year = {}
    p = REPO / f"data/{split}/aime-selection.json"
    if p.is_file():
        for e in json.loads(p.read_text())["episodes"]:
            aime_year[e["episode_id"]] = e["source_key"]["year"]
    fam, grp = zip(*(family_of(eid, t, dm_module, aime_year) for eid, t in zip(ids, texts)))
    if grouping == "template":
        grp = tuple(None if (g and g.startswith("aime:")) else g for g in grp)
    mult = {t: float(policy.tiers[t].budget_multiplier) for t in TIERS}
    return dict(ids=ids, texts=texts, X=X, Y=Y, score=truth_score, cost=truth_cost, family=list(fam), group=list(grp), mult=mult)


def make_folds(family, group, seed: int, k: int = K_FOLDS) -> np.ndarray:
    n = len(family)
    rng = random.Random(seed)
    fold = -np.ones(n, dtype=int)
    load = [0] * k
    grouped = {}
    for i, g in enumerate(group):
        if g is not None:
            grouped.setdefault(g, []).append(i)
    names = list(grouped)
    rng.shuffle(names)
    for g in sorted(names, key=lambda g: -len(grouped[g])):
        f = min(range(k), key=lambda f: (load[f], rng.random()))
        for i in grouped[g]:
            fold[i] = f
        load[f] += len(grouped[g])
    by_family = {}
    for i, (fam, g) in enumerate(zip(family, group)):
        if g is None:
            by_family.setdefault(fam, []).append(i)
    for fam in sorted(by_family):
        members = by_family[fam]
        rng.shuffle(members)
        start = rng.randrange(k)
        for j, i in enumerate(members):
            fold[i] = (start + j) % k
    assert (fold >= 0).all()
    return fold


def fit_fold(X, Y, tr):
    Xt, Yt = X[tr], Y[tr]
    kf = slice(KFEAT_SLOT, KFEAT_SLOT + KFEAT_DIM)
    mu = Xt[:, kf].mean(axis=0)
    sd = Xt[:, kf].std(axis=0) + 1e-9
    Xz = Xt.copy()
    Xz[:, kf] = (Xt[:, kf] - mu) / sd
    W = fit_dual_ridge(Xz, Yt, LAMBDA)
    W[BIAS_SLOT, :] -= (mu / sd) @ W[kf, :]
    W[kf, :] = W[kf, :] / sd[:, None]
    gbar = float((Yt[:, 1] - Yt[:, 0]).mean())
    W[:, 1] = GAMMA * W[:, 1] + (1.0 - GAMMA) * W[:, 0]
    W[BIAS_SLOT, 1] += (1.0 - GAMMA) * gbar
    resid = Yt[:, 3:] - (Xt @ W)[:, 3:]
    return W, np.exp(resid).mean(axis=0), resid.std(axis=0)


def build_oof(data):
    """seed별 OOF 예측: S[seed] (n×3 점수), C[seed] (n×3 smear 보정 비용), SIG[seed] (n×3 fold σ), FOLD[seed]."""
    X, Y = data["X"], data["Y"]
    n = X.shape[0]
    S, C, SIG, FOLD = {}, {}, {}, {}
    for seed in SEEDS:
        fold = make_folds(data["family"], data["group"], seed)
        s = np.zeros((n, 3))
        c = np.zeros((n, 3))
        sg = np.zeros((n, 3))
        for f in range(K_FOLDS):
            te = fold == f
            W, smear, sigma = fit_fold(X, Y, ~te)
            raw = X[te] @ W
            s[te] = np.clip(raw[:, :3], 0.0, 1.0)
            c[te] = np.exp(np.clip(raw[:, 3:], -20.0, 5.0)) * smear[None, :]
            sg[te] = sigma[None, :]
        S[seed], C[seed], SIG[seed], FOLD[seed] = s, c, sg, fold
        print(f"  seed {seed}: fold sizes {np.bincount(fold).tolist()}", flush=True)
    return S, C, SIG, FOLD


# ---------------------------------------------------------------- 평가 (worker 전역)
G = {}


def _alloc(idx, seed, tier, guard_mask, guard_mode, cfg, beta=1.0, margin_override=None):
    S = G["S"][seed][idx].copy()
    C = G["C"][seed][idx]
    SG = G["SIG"][seed][idx]
    for mask, mode in zip(guard_mask, guard_mode):
        g = mask[idx]
        if mode == "K":
            S[g, 2] = S[g, 1]
        else:
            S[g, 1] = S[g, 0]
            S[g, 2] = S[g, 0]
    pess = np.exp(beta * SG)
    pess[:, 0] = 1.0
    Cp = C * pess
    preds = [
        {MODEL_IDS[0]: (float(S[i, 0]), float(Cp[i, 0])), MODEL_IDS[1]: (float(S[i, 1]), float(Cp[i, 1])), MODEL_IDS[2]: (float(S[i, 2]), float(Cp[i, 2]))}
        for i in range(len(idx))
    ]
    margin = margin_at(cfg, len(idx)) if margin_override is None else margin_override
    choice = allocate(preds, G["mult"][tier], margin)
    j = np.array([MODEL_IDS.index(c) for c in choice])
    ts, tc = G["score"][idx], G["cost"][idx]
    score = float(ts[np.arange(len(idx)), j].mean())
    ratio = float(tc[np.arange(len(idx)), j].sum() / tc[:, 0].sum())
    return score, ratio


def evaluate_tier_config(job):
    guard_name, tier, cfg = job
    masks = [G["guard_masks"][guard_name][k] for k in range(len(G["guard_masks"][guard_name]))]
    modes = G["guard_modes"][guard_name]
    lim = G["mult"][tier]
    out = {"guard": guard_name, "tier": tier, "cfg": cfg_name(cfg), "cfg_raw": cfg}
    # fold 게이트 + vpCV
    fold_rows = []
    for seed in SEEDS:
        for f in range(K_FOLDS):
            idx = np.where(G["FOLD"][seed] == f)[0]
            score, ratio = _alloc(idx, seed, tier, masks, modes, cfg)
            fold_rows.append((score, ratio))
    out["fold_worst_ratio"] = max(r for _, r in fold_rows)
    out["fold_over"] = sum(r > lim for _, r in fold_rows)
    out["fold_gate"] = all(r <= lim * FOLD_GATE[tier] for _, r in fold_rows)
    out["vpcv"] = statistics.mean(s if r <= lim else 0.0 for s, r in fold_rows)
    out["rawcv"] = statistics.mean(s for s, _ in fold_rows)
    out["fold_mean_ratio"] = statistics.mean(r for _, r in fold_rows)
    # 부분집합 게이트
    sub = {}
    for n in SUBSAMPLE_SIZES:
        rows = [_alloc(idx, seed, tier, masks, modes, cfg) for seed, idx in G["subsamples"][n]]
        ratios = [r for _, r in rows]
        sub[n] = {
            "pass": sum(r <= lim for r in ratios) / len(ratios),
            "mean_ratio": statistics.mean(ratios),
            "max_ratio": max(ratios),
            "p99_ratio": sorted(ratios)[int(0.99 * (len(ratios) - 1))],
            "mean_score": statistics.mean(s for s, _ in rows),
        }
    out["subsample"] = sub
    out["subsample_gate"] = all(sub[n]["pass"] >= SUBSAMPLE_MIN_PASS[n] for n in SUBSAMPLE_SIZES)
    # 전체 OOF 풀 (n=1760) — seed 3개
    full = [_alloc(np.arange(len(G["score"])), seed, tier, masks, modes, cfg) for seed in SEEDS]
    out["full_pool"] = {"max_ratio": max(r for _, r in full), "mean_score": statistics.mean(s for s, _ in full)}
    # 출처 편향 게이트
    fam_rows = {}
    for fam, idx in G["family_idx"].items():
        rs = [_alloc(idx, seed, tier, masks, modes, cfg) for seed in SEEDS]
        fam_rows[fam] = {"n": int(len(idx)), "max_ratio": max(r for _, r in rs), "mean_score": statistics.mean(s for s, _ in rs)}
    out["family_only"] = fam_rows
    half = [_alloc(idx, seed, tier, masks, modes, cfg) for seed, idx in G["half_family"]]
    out["half_family_pass"] = sum(r <= lim for _, r in half) / len(half)
    out["family_worst_ratio"] = max(v["max_ratio"] for v in fam_rows.values())
    deep_margin = cfg[1] if cfg[1] is not None else cfg[0]
    fam_deep = {}
    for fam, idx in G["family_idx"].items():
        rs = [_alloc(idx, seed, tier, masks, modes, cfg, margin_override=deep_margin) for seed in SEEDS]
        fam_deep[fam] = {"n": int(len(idx)), "max_ratio": max(r for _, r in rs), "mean_score": statistics.mean(s for s, _ in rs)}
    out["family_deep"] = fam_deep
    out["family_deep_worst_ratio"] = max(v["max_ratio"] for v in fam_deep.values())
    out["family_deep_gate"] = out["family_deep_worst_ratio"] <= lim * FAMILY_GATE
    out["family_deep_gate_relaxed"] = out["family_deep_worst_ratio"] <= lim
    out["family_gate"] = out["family_worst_ratio"] <= lim * FAMILY_GATE and out["half_family_pass"] >= HALF_FAMILY_MIN_PASS
    out["deploy_worst_ratio"] = sub[880]["max_ratio"]
    out["deploy_gate"] = out["deploy_worst_ratio"] <= lim * DEPLOY_GATE
    # 완충 1.0 버전 — 후보 전원이 못 넘는 게이트를 완화할 때 쓴다
    out["fold_gate_relaxed"] = out["fold_over"] == 0
    out["family_gate_relaxed"] = out["family_worst_ratio"] <= lim and out["half_family_pass"] >= HALF_FAMILY_MIN_PASS
    out["deploy_gate_relaxed"] = out["deploy_worst_ratio"] <= lim
    out["all_gates"] = out["fold_gate"] and out["subsample_gate"] and out["family_gate"] and out["deploy_gate"] and out["family_deep_gate"]
    return out


def guard_stats(data, name):
    """가드가 걸리는 train 문항의 L/M/K 실측 요약 — 가드가 무엇을 포기하는지 보여준다."""
    masks = G["guard_masks"][name]
    any_mask = np.zeros(len(data["ids"]), dtype=bool)
    for m in masks:
        any_mask |= m
    idx = np.where(any_mask)[0]
    if len(idx) == 0:
        return {"n": 0}
    sc, co = data["score"][idx], data["cost"][idx]
    return {
        "n": int(len(idx)),
        "mean_score_LMK": [round(float(sc[:, j].mean()), 3) for j in range(3)],
        "sum_cost_K": round(float(co[:, 2].sum()), 2),
        "max_cost_K": round(float(co[:, 2].max()), 2),
        "sum_cost_M": round(float(co[:, 1].sum()), 3),
        "max_cost_M": round(float(co[:, 1].max()), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="train 전용 교차검증 게이트")
    parser.add_argument("--input", type=Path, default=REPO / "data/materialized/train/inputs.json")
    parser.add_argument("--outcomes", type=Path, default=REPO / "data/train/outcomes.json")
    parser.add_argument("--report", type=Path, default=REPO / "build/cv-gate-report.json")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--grouping", choices=("template", "source-holdout"), default="template")
    parser.add_argument("--guards", default=",".join(GUARD_VARIANTS_ALL), help="쉼표로 구분한 가드 변형 이름")
    args = parser.parse_args()
    global GUARD_VARIANTS
    GUARD_VARIANTS = {k: GUARD_VARIANTS_ALL[k] for k in args.guards.split(",")}
    t0 = time.time()
    print("[1/4] 자료·특징·그룹 구축", flush=True)
    data = load_all(args.input, args.outcomes, args.grouping)
    print(f"  그룹화: {args.grouping}", flush=True)
    n = len(data["ids"])
    fam_count = {}
    for f in data["family"]:
        fam_count[f] = fam_count.get(f, 0) + 1
    print("  가족:", dict(sorted(fam_count.items(), key=lambda kv: -kv[1])), flush=True)
    print(f"  그룹 수: {len(set(g for g in data['group'] if g))} ({'DM 모듈' if args.grouping == 'template' else 'DM 모듈·AIME 연도'})", flush=True)
    print("[2/4] OOF 예측 (5-fold × 3 seed 재적합)", flush=True)
    S, C, SIG, FOLD = build_oof(data)
    G.update(S=S, C=C, SIG=SIG, FOLD=FOLD, score=data["score"], cost=data["cost"], mult=data["mult"])
    # 가드 마스크
    G["guard_masks"], G["guard_modes"] = {}, {}
    for name, items in GUARD_VARIANTS.items():
        G["guard_masks"][name] = [np.array([fn(t) for t in data["texts"]], dtype=bool) for fn, _ in items]
        G["guard_modes"][name] = [mode for _, mode in items]
    for name in GUARD_VARIANTS:
        print(f"  {name:26s} {guard_stats(data, name)}", flush=True)
    # 공통 난수 부분집합
    rng = random.Random(2026)
    G["subsamples"] = {sz: [(SEEDS[r % len(SEEDS)], np.array(sorted(rng.sample(range(n), sz)))) for r in range(SUBSAMPLE_REPS)] for sz in SUBSAMPLE_SIZES}
    fam_idx = {}
    for i, f in enumerate(data["family"]):
        fam_idx.setdefault(f, []).append(i)
    G["family_idx"] = {f: np.array(v) for f, v in fam_idx.items() if len(v) >= 40}
    half = []
    for f, members in G["family_idx"].items():
        others = [i for i in range(n) if data["family"][i] != f]
        for r in range(HALF_FAMILY_REPS):
            k = min(HALF_FAMILY_N // 2, len(members))
            pick = rng.sample(list(members), k) + rng.sample(others, HALF_FAMILY_N - k)
            half.append((SEEDS[r % len(SEEDS)], np.array(sorted(pick))))
    G["half_family"] = half
    print(f"[3/4] 후보 평가 ({time.time()-t0:.0f}s 경과)", flush=True)
    jobs = []
    for gname in GUARD_VARIANTS:
        for cfg in FAST_GRID:
            jobs.append((gname, "fast", cfg))
        for cfg in BALANCED_GRID:
            jobs.append((gname, "balanced", cfg))
        for cfg in PREMIUM_GRID:
            jobs.append((gname, "premium", cfg))
    print(f"  작업 {len(jobs)}건, worker {args.workers}", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(evaluate_tier_config, jobs, chunksize=2)
    print(f"[4/4] 선택 ({time.time()-t0:.0f}s 경과)", flush=True)
    by = {}
    for r in results:
        by.setdefault(r["guard"], {}).setdefault(r["tier"], []).append(r)
    GATES = ("fold_gate", "subsample_gate", "family_gate", "deploy_gate", "family_deep_gate")

    def select(rows):
        """게이트 전부 통과 → vpCV 최대 (동률이면 n=880 점수). 전원이 못 넘는 게이트만 완충 1.0으로 완화."""
        amended = []
        active = {g: g for g in GATES}
        for g in GATES:
            if g != "subsample_gate" and not any(r[g] for r in rows):
                active[g] = g + "_relaxed"
                amended.append(g)
        ok = [r for r in rows if all(r[active[g]] for g in GATES)]
        best = max(ok, key=lambda r: (r["vpcv"], r["subsample"][880]["mean_score"])) if ok else None
        return best, amended

    selection = {}
    for gname, tiers in by.items():
        sel, amended_all = {}, {}
        for tier, rows in tiers.items():
            sel[tier], amended_all[tier] = select(rows)
        final = sum(WEIGHTS[t] * sel[t]["vpcv"] for t in TIERS) if all(sel[t] for t in TIERS) else None
        selection[gname] = {
            "tiers": {t: (sel[t]["cfg"] if sel[t] else None) for t in TIERS},
            "vpcv_final": final,
            "amended_gates": amended_all,
        }
    # 현행 구성의 수치 (비교용) + 현행 대비 지배(모든 안전 지표가 같거나 낫고 vpCV도 같거나 높음)
    current = {}
    for gname in GUARD_VARIANTS:
        rows = {t: next(r for r in by[gname][t] if r["cfg_raw"] == CURRENT[t]) for t in TIERS}
        current[gname] = {t: rows[t] for t in TIERS}
    base_name = "G0-current" if "G0-current" in GUARD_VARIANTS else next(iter(GUARD_VARIANTS))
    base = current[base_name]
    for r in results:
        c = base[r["tier"]]
        r["dominates_current"] = (
            r["vpcv"] >= c["vpcv"] - 1e-12
            and r["fold_over"] <= c["fold_over"]
            and r["fold_worst_ratio"] <= c["fold_worst_ratio"] + 1e-12
            and all(r["subsample"][n]["pass"] >= c["subsample"][n]["pass"] for n in SUBSAMPLE_SIZES)
            and r["half_family_pass"] >= c["half_family_pass"]
            and max(v["max_ratio"] for v in r["family_only"].values()) <= max(v["max_ratio"] for v in c["family_only"].values()) + 1e-12
        )
    dominant = {}
    for gname, tiers in by.items():
        dominant[gname] = {}
        for tier, rows in tiers.items():
            ok = [r for r in rows if r["dominates_current"]]
            dominant[gname][tier] = sorted(ok, key=lambda r: -r["vpcv"])[:3]
    report = {
        "seeds": SEEDS, "folds": K_FOLDS, "gates": {"fold": FOLD_GATE, "subsample_min_pass": SUBSAMPLE_MIN_PASS, "family": FAMILY_GATE, "half_family_min_pass": HALF_FAMILY_MIN_PASS},
        "families": fam_count, "guard_stats": {g: guard_stats(data, g) for g in GUARD_VARIANTS},
        "results": results, "selection": selection, "grouping": args.grouping,
        "guard_definitions": guard_definitions(),
        "current_config_raw": CURRENT,
        "dominates_current": {g: {t: [r["cfg"] for r in dominant[g][t]] for t in TIERS} for g in GUARD_VARIANTS},
        "current_config": {g: {t: {k: v for k, v in current[g][t].items() if k != "cfg_raw"} for t in TIERS} for g in GUARD_VARIANTS},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # 요약 출력
    def line(r):
        s = r["subsample"]
        return (f"{r['cfg']:16s} vpCV={r['vpcv']:.4f} raw={r['rawcv']:.4f} foldworst={r['fold_worst_ratio']:.3f} over={r['fold_over']:2d} "
                f"| pass n100={s[100]['pass']*100:5.1f} n200={s[200]['pass']*100:5.1f} n400={s[400]['pass']*100:5.1f} n800={s[800]['pass']*100:5.1f} n880={s[880]['pass']*100:5.1f} "
                f"| n880 ratio={s[880]['mean_ratio']:.3f} max={s[880]['max_ratio']:.3f} score={s[880]['mean_score']:.4f} | fam={'ok' if r['family_gate'] else 'FAIL'}({r['family_worst_ratio']:.3f}) half={r['half_family_pass']*100:.0f}% dep={'ok' if r['deploy_gate'] else 'FAIL'}({r['deploy_worst_ratio']:.3f}) famD={'ok' if r['family_deep_gate'] else 'FAIL'}({r['family_deep_worst_ratio']:.3f}) | {'PASS' if r['all_gates'] else 'fail'}{' ≥cur' if r.get('dominates_current') else ''}")
    for gname in GUARD_VARIANTS:
        print(f"\n=== {gname} ===")
        for tier in TIERS:
            print(f" [{tier}] limit {data['mult'][tier]}")
            for r in sorted(by[gname][tier], key=lambda r: -r["vpcv"]):
                print("   " + line(r))
    print(f"\n=== 선택 (게이트 통과 중 vpCV 최대; 현행 가드 k_guard@{guard_definitions()['current_k_guard_source_sha256'][:8]}) ===")
    for gname, s in selection.items():
        print(f"  {gname:26s} {s['tiers']}  vpCV_final={s['vpcv_final'] if s['vpcv_final'] is None else round(s['vpcv_final'], 4)}  완화한 게이트={s['amended_gates']}")
    print(f"\n=== 현행 margin 구성({base_name}) 대비 전 지표 지배 후보 (등급별 상위 3) ===")
    for gname in GUARD_VARIANTS:
        print(f"  {gname:26s} " + " | ".join(f"{t}: {[r['cfg'] for r in dominant[gname][t]]}" for t in TIERS))
    cur = current[base_name]
    print(f"  현행 margin 구성({base_name}, 현재 k_guard 기준) vpCV_final={sum(WEIGHTS[t]*cur[t]['vpcv'] for t in TIERS):.4f} gates={{{', '.join(t+':'+('PASS' if cur[t]['all_gates'] else 'fail') for t in TIERS)}}}")
    print(f"보고서: {args.report} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
