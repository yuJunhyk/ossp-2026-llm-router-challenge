# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""C2 게이트 — 한국어 형태소 근사 어간 채널 추가 (train 전용, V15-GATES.md 사전 등록).

기존 u(단어)/b(바이그램)/c(문자 4그램) 해시 채널은 그대로 두고, 한글 어절에서
닫힌 조사 목록을 최장 일치로 벗겨낸 어간을 새 salt "s" 채널로 추가한다.
어간 채널은 별도 L2 정규화 후 가산하므로 기존 채널의 값은 한 비트도 변하지 않는다.
"학교에서/학교를/학교가"가 같은 버킷에 모이는 것이 노리는 효과이며,
korean-heavy(181문항 규모) 국한 신호라 기대값은 낮게 등록돼 있다.

게이트: 동일 fold 짝지음 ΔCV ≥ +0.004. 스크리닝 전 기준선 0.6553 재현 필수.
산출물은 build/v15/ 전용, 원본 저장소 무접촉.
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
    calibrate,
    episode_text,
    group_fold_ids,
    linear_candidate,
    realized,
    rows_with_pessimism,
    stress_masks,
    template_group_keys,
)
from os2_features import NUMERIC_DIM, TOTAL_DIM, _WORD_RE, _bucket, extract_sparse
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
# v1.3 고정 캘리브레이션 (진단 전용 — V15-GATES.md)
V13_CONFIG = {"fast": (0.5, 0.90), "balanced": (0.5, 0.96), "premium": (1.0, 0.80)}

# 닫힌 조사 목록 — 최장 일치 우선. 어미 활용은 다루지 않는다(과절단 방지).
_JOSA = sorted(
    [
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "과",
        "와",
        "의",
        "도",
        "만",
        "에",
        "에서",
        "에게",
        "께서",
        "께",
        "한테",
        "한테서",
        "에게서",
        "같이",
        "처럼",
        "만큼",
        "보다",
        "부터",
        "까지",
        "마다",
        "조차",
        "마저",
        "밖에",
        "뿐",
        "대로",
        "로",
        "으로",
        "로서",
        "으로서",
        "로써",
        "으로써",
        "로부터",
        "으로부터",
        "라도",
        "이라도",
        "나",
        "이나",
        "나마",
        "이나마",
        "든지",
        "이든지",
        "라든지",
        "랑",
        "이랑",
        "하고",
        "커녕",
        "은커녕",
        "는커녕",
        "에다",
        "에다가",
        "에서는",
        "에서의",
        "에의",
        "와의",
        "과의",
        "이란",
        "란",
        "이라는",
        "라는",
        "이며",
        "이자",
    ],
    key=len,
    reverse=True,
)


def _stem(token: str) -> str | None:
    """한글 어절에서 조사를 벗긴 어간. 못 벗기거나 2자 미만이면 None."""
    for josa in _JOSA:
        if token.endswith(josa):
            stem = token[: -len(josa)]
            if len(stem) >= 2:
                return stem
            return None
    return None


def extract_sparse_stem(text: str) -> dict[int, float]:
    """기존 extract_sparse 결과에 salt "s" 어간 채널을 가산."""
    feats = dict(extract_sparse(text))
    clipped = text[:20000]
    words = _WORD_RE.findall(clipped)[:3000]
    counts: dict[int, float] = {}
    for w in words:
        if not ("가" <= w[0] <= "힣"):
            continue
        stem = _stem(w)
        if stem is None:
            continue
        idx, sign = _bucket(stem, "s")
        counts[idx] = counts.get(idx, 0.0) + sign
    if counts:
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm > 0:
            for idx, val in counts.items():
                key = NUMERIC_DIM + idx
                feats[key] = feats.get(key, 0.0) + val / norm
    return feats


def build_matrix(texts, extractor):
    X = np.zeros((len(texts), TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extractor(t).items():
            X[i, idx] = val
    return X


def fold_weighted_scores(bundles, tier_cfgs, truth, policy):
    """15개 fold 각각의 (0.4·fast + 0.3·bal + 0.3·prem) 실현 점수와 예산 위반 수."""
    true_score, true_cost = truth["score"], truth["cost"]
    per_fold = []
    violations = 0
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

    # 어간 채널 커버리지 리포트
    n_stemmed = sum(1 for t in texts if extract_sparse_stem(t) != extract_sparse(t))
    print(f"어간 채널이 특징을 바꾼 문항: {n_stemmed}/{n}")

    print("[1/3] 기준선 재현 (extract_sparse, λ=10)", flush=True)
    X_base = build_matrix(texts, extract_sparse)
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

    print("[2/3] 변형 평가 (extract_sparse_stem, λ=10)", flush=True)
    X_stem = build_matrix(texts, extract_sparse_stem)
    b_stem, oof_stem = linear_candidate(X_stem, Y, fold_sets, LAM)
    res_stem = calibrate(
        "linear+stem", attach_light_base(b_stem), oof_stem, masks, truth, policy
    )
    delta = res_stem["weighted"] - res_base["weighted"]
    print(f"  stem weighted_cv={res_stem['weighted']:.4f}  Δ={delta:+.4f}")
    for tier, cfg in res_stem["tiers"].items():
        if cfg:
            print(
                f"  {tier:9s} β={cfg['beta']:.2f} margin={cfg['margin']:.2f} "
                f"cv={cfg['cv_score']:.4f} max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']}"
            )

    print("[3/3] 진단 — paired per-fold Δ + v1.3 고정 캘리브레이션", flush=True)
    # per-fold Δ는 각자의 채택 캘리브레이션에서 측정 (게이트와 같은 조건)
    base_cfg = {
        t: (res_base["tiers"][t]["beta"], res_base["tiers"][t]["margin"]) for t in TIERS
    }
    stem_cfg = {
        t: (res_stem["tiers"][t]["beta"], res_stem["tiers"][t]["margin"]) for t in TIERS
    }
    pf_base, _ = fold_weighted_scores(b_base, base_cfg, truth, policy)
    pf_stem, _ = fold_weighted_scores(b_stem, stem_cfg, truth, policy)
    pf_delta = pf_stem - pf_base
    pos = int((pf_delta > 0).sum())
    print(f"  per-fold Δ>0: {pos}/15 (참고 지표 ≥10)")

    # 진단: v1.3 고정 (β,margin)에서의 Δ — 캘리브레이션 이동분 분해
    fx_base, vio_b = fold_weighted_scores(b_base, V13_CONFIG, truth, policy)
    fx_stem, vio_s = fold_weighted_scores(b_stem, V13_CONFIG, truth, policy)
    fixed_delta = float(fx_stem.mean() - fx_base.mean())
    print(
        f"  v1.3 고정 캘리브레이션 Δ={fixed_delta:+.4f} (예산 위반 base {vio_b}, stem {vio_s})"
    )

    verdict = delta >= GATE_DELTA
    print(f"C2 게이트: {'통과' if verdict else '미달 → 기각'} (문턱 +{GATE_DELTA})")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    out = REPO / "build/v15/stem-channel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate": "C2 stem-channel",
                "git": git_hash,
                "baseline_weighted": res_base["weighted"],
                "variant_weighted": res_stem["weighted"],
                "delta": delta,
                "per_fold_delta": pf_delta.tolist(),
                "per_fold_positive": pos,
                "fixed_v13_delta": fixed_delta,
                "episodes_with_stem_features": n_stemmed,
                "pass": bool(verdict),
                "tiers": res_stem["tiers"],
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
