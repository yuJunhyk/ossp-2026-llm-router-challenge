# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""예측기 재대결 하니스 — ens(ridge+GBM diff) vs 선형(dual ridge). train 전용.

배경: v1.2까지의 모델 선택·앙상블 가중치·배포 버퍼는 Dev 채점이 개입해
낙관 편향이 있다(감사 결과). 여기서는 Dev를 전혀 사용하지 않고, 동일한
프로토콜 아래에서 두 예측기를 공정하게 재대결시켜 배포 예측기를 다시
선택한다.

프로토콜 (모든 선택은 train으로만):
- fold: 템플릿 그룹 단위 5-fold × fold-seed 3개 = 15 fold.
  정규화한 25자 접두사를 공유하는 문항(2개 이상)은 같은 fold에 묶어
  순번 modulo 분할의 템플릿 누수를 차단한다.
- 배분 정책: analysis/os2_policy.allocate (볼록껍질 그리디) — 두 후보 동일.
  비용 비관 계수 β(log-잔차 σ의 배수, light 제외)와 margin을 함께 격자 탐색.
- 캘리브레이션 제약: tier별 (β, margin)은
  (a) 15개 fold 전부에서 실현 예산비 ≤ 한도 × SAFETY_CAP{0.90, 0.95, 0.95}
  (b) 유형 편향 부분집합(9종) OOF 실현 예산비 ≤ 한도 × 0.98
  을 만족해야 하며, 만족하는 설정 중 CV 평균 점수 최대를 채택한다.
- 최종 비교: weighted CV = 0.4·fast + 0.3·balanced + 0.3·premium.
- 후보 내부 변형(선형 λ, ens 앙상블 가중 w)도 같은 기준으로 재선택한다
  (v1.2 값 이어받기 금지).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np
import lightgbm as lgb

from train_hash_regex import _fit_ridge, _predict_ridge, _training_matrix
from train_ens import (
    GBM_PARAMS,
    HASH_BINS,
    RIDGE_ALPHA,
    TARGET_NAMES,
    to_diff,
    reconstruct,
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

CV_SEEDS = (11, 22, 33)
FOLDS = 5
GBM_SEEDS = (7, 8, 9)
ROUND_GRID = (100, 200, 400, 800, 1200, 1600)
LINEAR_LAMBDAS = (1.0, 3.0, 10.0)
ENS_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS = (0.0, 0.5, 1.0, 1.5)
MARGINS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96)
SAFETY_CAP = {"fast": 0.90, "balanced": 0.95, "premium": 0.95}
STRESS_CAP = 0.98
GROUP_PREFIX = 25


def episode_text(episode) -> str:
    if episode.prompt is not None:
        return episode.prompt
    parts = []
    for message in episode.messages or ():
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def template_group_keys(texts: list[str]) -> list[str]:
    """같은 템플릿 계열(정규화 25자 접두사 공유, 2개 이상)을 한 그룹으로."""
    normed = [re.sub(r"\s+", " ", t.strip())[:GROUP_PREFIX] for t in texts]
    counts = Counter(normed)
    return [k if counts[k] >= 2 else f"~단독{i}" for i, k in enumerate(normed)]


def group_fold_ids(keys: list[str], seed: int, folds: int = FOLDS) -> np.ndarray:
    """그룹을 통째로 fold에 배정 (크기 균형 그리디, seed로 순서 셔플)."""
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(key, []).append(i)
    order = sorted(members)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    sizes = [0] * folds
    fold_of = np.empty(len(keys), dtype=np.int64)
    for key in order:
        target = min(range(folds), key=lambda f: sizes[f])
        for i in members[key]:
            fold_of[i] = target
        sizes[target] += len(members[key])
    return fold_of


def stress_masks(texts: list[str]) -> dict[str, np.ndarray]:
    """유형 편향 부분집합 (독립 재구현 세션 train_router.py와 동일 정의)."""
    hangul = np.array(
        [
            sum(1 for c in t[:2000] if "가" <= c <= "힣") / max(1, len(t[:2000]))
            for t in texts
        ]
    )
    n_chars = np.array([len(t) for t in texts])
    mc = np.array([1 if re.search(r"(?:^|\n|\s)[A-E]\.\s", t) else 0 for t in texts])
    mathy = np.array(
        [1 if re.search(r"[=+\-*/^]|\bpolynomial\b|\bLet\b", t) else 0 for t in texts]
    )
    code = np.array(
        [1 if ("def " in t or "```" in t or "assert" in t) else 0 for t in texts]
    )
    masks = {
        "korean-heavy": hangul > 0.1,
        "english-only": hangul == 0,
        "short": n_chars < 300,
        "long": n_chars > 1500,
        "mc": mc == 1,
        "no-mc": mc == 0,
        "mathy": mathy == 1,
        "non-mathy": mathy == 0,
        "codey": code == 1,
    }
    return {k: np.where(v)[0] for k, v in masks.items() if v.sum() >= 50}


def fit_dual_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """w = X^T (XX^T + lam I)^{-1} Y (독립 재구현 세션 train_router.py 이식)."""
    n = X.shape[0]
    K = X @ X.T
    K[np.diag_indices(n)] += lam
    alpha = np.linalg.solve(K, Y)
    return X.T @ alpha


def cost_stats(
    pred_log: np.ndarray, true_log: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """log-비용 잔차의 smear(기대값 보정)와 σ (모델별)."""
    resid = true_log - pred_log
    return np.exp(resid).mean(axis=0), resid.std(axis=0)


def rows_with_pessimism(
    pred: np.ndarray, smear: np.ndarray, sigma: np.ndarray, beta: float
) -> list[dict[str, tuple[float, float]]]:
    """(n,6) 예측 → allocate 입력. 비용은 smear 보정 후 β·σ 비관(light 제외)."""
    scores = np.clip(pred[:, :3], 0.0, 1.0)
    cost = np.exp(np.clip(pred[:, 3:], -20.0, 5.0)) * smear[None, :]
    pess = np.array([1.0, math.exp(beta * sigma[1]), math.exp(beta * sigma[2])])
    cost = cost * pess[None, :]
    # 비용 단조 보정 (예측 역전 방지)
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


def realized(idx, choice, true_score, true_cost, light_base):
    score = sum(true_score[i][c] for i, c in zip(idx, choice)) / len(idx)
    used = sum(true_cost[i][c] for i, c in zip(idx, choice)) / light_base
    return score, used


def calibrate(
    name: str,
    bundles: list[dict],
    oof_by_seed: list[dict],
    masks: dict[str, np.ndarray],
    truth: dict,
    policy,
) -> dict:
    """(β, margin) 격자 → tier별 feasible 중 CV 점수 최대. weighted 반환."""
    true_score, true_cost = truth["score"], truth["cost"]
    light_costs = truth["light_costs"]
    result = {"name": name, "tiers": {}, "weighted": 0.0}
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        weight = float(policy.tiers[tier].weight)
        feasible = []
        for beta in BETAS:
            fold_rows = [
                rows_with_pessimism(b["pred"], b["smear"], b["sigma"], beta)
                for b in bundles
            ]
            oof_rows = [
                rows_with_pessimism(o["pred"], o["smear"], o["sigma"], beta)
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


def linear_candidate(X, Y, fold_sets, lam):
    """선형 dual ridge: fold별 예측 번들 + seed별 전체 OOF."""
    bundles, oof_by_seed = [], []
    for fold_of in fold_sets:
        raw_full = np.zeros_like(Y)
        smear_acc, sigma_acc = [], []
        for k in range(FOLDS):
            va = fold_of == k
            W = fit_dual_ridge(X[~va], Y[~va], lam)
            pred_tr = X[~va] @ W
            smear, sigma = cost_stats(pred_tr[:, 3:], Y[~va][:, 3:])
            pred_va = X[va] @ W
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


def ens_fold_predictions(X, Y, diff, fold_sets):
    """ens 원자재: fold별 ridge 예측 + GBM staged 예측 (라운드 격자 전부)."""
    per_fold = []
    for fold_of in fold_sets:
        for k in range(FOLDS):
            va = fold_of == k
            tr = ~va
            m, s, b, c = _fit_ridge(X[tr], Y[tr], RIDGE_ALPHA)
            ridge_va = _predict_ridge(X[va], m, s, b, c)
            ridge_tr = _predict_ridge(X[tr], m, s, b, c)
            gbm_va = np.zeros((len(ROUND_GRID), int(va.sum()), diff.shape[1]))
            gbm_tr = np.zeros((len(ROUND_GRID), int(tr.sum()), diff.shape[1]))
            for t in range(diff.shape[1]):
                for seed in GBM_SEEDS:
                    ds = lgb.Dataset(X[tr], label=diff[tr][:, t])
                    booster = lgb.train(
                        dict(GBM_PARAMS, seed=seed), ds, num_boost_round=ROUND_GRID[-1]
                    )
                    for ri, r in enumerate(ROUND_GRID):
                        gbm_va[ri, :, t] += booster.predict(
                            X[va], num_iteration=r
                        ) / len(GBM_SEEDS)
                        gbm_tr[ri, :, t] += booster.predict(
                            X[tr], num_iteration=r
                        ) / len(GBM_SEEDS)
            per_fold.append(
                {
                    "va_idx": np.where(va)[0],
                    "tr_idx": np.where(tr)[0],
                    "ridge_va": ridge_va,
                    "ridge_tr": ridge_tr,
                    "gbm_va": gbm_va,
                    "gbm_tr": gbm_tr,
                }
            )
            print(f"  ens fold {len(per_fold)}/15 완료", flush=True)
    return per_fold


def select_ens_rounds(per_fold, diff):
    """타깃별 라운드: 15 fold 합산 OOF MSE 최소 (train 전용)."""
    chosen = []
    for t in range(diff.shape[1]):
        errs = []
        for ri in range(len(ROUND_GRID)):
            sse = 0.0
            for f in per_fold:
                resid = f["gbm_va"][ri, :, t] - diff[f["va_idx"], t]
                sse += float(resid @ resid)
            errs.append(sse)
        chosen.append(ROUND_GRID[int(np.argmin(errs))])
    return chosen


def ens_candidate(per_fold, Y, diff, rounds, weight_w):
    """선택된 라운드·가중치로 fold 번들과 seed별 전체 OOF 구성."""
    round_index = {r: i for i, r in enumerate(ROUND_GRID)}
    bundles = []
    n = Y.shape[0]
    per_seed_pred = [np.zeros_like(Y) for _ in CV_SEEDS]
    per_seed_smear = [[] for _ in CV_SEEDS]
    per_seed_sigma = [[] for _ in CV_SEEDS]
    for fi, f in enumerate(per_fold):
        seed_slot = fi // FOLDS
        gbm_va = np.stack(
            [f["gbm_va"][round_index[rounds[t]], :, t] for t in range(diff.shape[1])],
            axis=1,
        )
        gbm_tr = np.stack(
            [f["gbm_tr"][round_index[rounds[t]], :, t] for t in range(diff.shape[1])],
            axis=1,
        )
        pred_va = (1 - weight_w) * f["ridge_va"] + weight_w * reconstruct(gbm_va)
        pred_tr = (1 - weight_w) * f["ridge_tr"] + weight_w * reconstruct(gbm_tr)
        smear, sigma = cost_stats(pred_tr[:, 3:], Y[f["tr_idx"]][:, 3:])
        bundles.append(
            {"idx": f["va_idx"], "pred": pred_va, "smear": smear, "sigma": sigma}
        )
        per_seed_pred[seed_slot][f["va_idx"]] = pred_va
        per_seed_smear[seed_slot].append(smear)
        per_seed_sigma[seed_slot].append(sigma)
    oof_by_seed = [
        {
            "pred": per_seed_pred[si],
            "smear": np.mean(per_seed_smear[si], axis=0),
            "sigma": np.mean(per_seed_sigma[si], axis=0),
        }
        for si in range(len(CV_SEEDS))
    ]
    return bundles, oof_by_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="예측기 재대결 (train 전용)")
    parser.add_argument(
        "--input", type=Path, default=REPO / "data/materialized/train/inputs.json"
    )
    parser.add_argument(
        "--outcomes", type=Path, default=REPO / "data/train/outcomes.json"
    )
    parser.add_argument(
        "--report", type=Path, default=REPO / "build/rematch-report.json"
    )
    parser.add_argument("--smoke", action="store_true", help="축소 검증 실행")
    args = parser.parse_args()

    global CV_SEEDS, ENS_WEIGHTS, LINEAR_LAMBDAS
    if args.smoke:
        CV_SEEDS = (11,)
        ENS_WEIGHTS = (0.5,)
        LINEAR_LAMBDAS = (1.0,)

    policy = load_bundled_policy()
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    texts = [episode_text(ep) for ep in inputs.episodes]
    n = len(texts)

    # 실제 score·비용 (채점기와 동일식)
    rates = {m: policy.models[m] for m in MODEL_IDS}
    truth_map = {}
    for oc in outcomes.outcomes:
        r = rates[oc.model_id]
        cost = (
            float(r.fixed_cost)
            + (
                oc.input_tokens * float(r.input_token_rate)
                + oc.output_tokens * float(r.output_token_rate)
            )
            / policy.token_unit
        )
        truth_map[(oc.episode_id, oc.model_id)] = (float(oc.score), cost)
    episode_ids = [ep.episode_id for ep in inputs.episodes]
    true_score = [{m: truth_map[(eid, m)][0] for m in MODEL_IDS} for eid in episode_ids]
    true_cost = [{m: truth_map[(eid, m)][1] for m in MODEL_IDS} for eid in episode_ids]
    light_costs = [c[MODEL_IDS[0]] for c in true_cost]
    truth = {"score": true_score, "cost": true_cost, "light_costs": light_costs}

    # 그룹 fold (템플릿 누수 차단)
    keys = template_group_keys(texts)
    grouped = sum(1 for k in keys if not k.startswith("~단독"))
    fold_sets = [group_fold_ids(keys, seed) for seed in CV_SEEDS]
    print(f"episodes={n}, 템플릿 그룹 소속={grouped}, fold-seed={CV_SEEDS}")

    masks = stress_masks(texts)
    print(f"stress subsets: {sorted((k, len(v)) for k, v in masks.items())}")

    # fold별 light 실제 기준비용
    def attach_light_base(bundles):
        for b in bundles:
            b["light_base"] = sum(light_costs[i] for i in b["idx"])
        return bundles

    results = []

    # ── 후보 A: 선형 dual ridge (특징: 수치 40 + crc32 8192)
    print(f"[선형] 특징 행렬 구축 (dim={TOTAL_DIM}) ...", flush=True)
    X_lin = np.zeros((n, TOTAL_DIM))
    for i, t in enumerate(texts):
        for idx, val in extract_sparse(t).items():
            X_lin[i, idx] = val
    Y_lin = np.zeros((n, 6))
    for i in range(n):
        for j, m in enumerate(MODEL_IDS):
            Y_lin[i, j] = true_score[i][m]
            Y_lin[i, 3 + j] = math.log(max(true_cost[i][m], 1e-9))
    for lam in LINEAR_LAMBDAS:
        print(f"[선형] λ={lam} CV ...", flush=True)
        bundles, oof = linear_candidate(X_lin, Y_lin, fold_sets, lam)
        res = calibrate(
            f"linear(λ={lam})", attach_light_base(bundles), oof, masks, truth, policy
        )
        results.append(res)
        print(f"  weighted_cv={res['weighted']:.4f}")

    # ── 후보 B: ens (ridge α=10 + GBM diff, 특징: hash_regex 270)
    print("[ens] 특징 행렬 구축 ...", flush=True)
    X_ens, Y_ens = _training_matrix(inputs, outcomes, policy, HASH_BINS)
    diff = to_diff(Y_ens)
    print("[ens] fold별 ridge + GBM staged 예측 (시간 소요) ...", flush=True)
    per_fold = ens_fold_predictions(X_ens, Y_ens, diff, fold_sets)
    rounds = select_ens_rounds(per_fold, diff)
    print(f"[ens] 라운드 선택 (그룹 OOF MSE): {dict(zip(TARGET_NAMES, rounds))}")
    for w in ENS_WEIGHTS:
        print(f"[ens] w={w} CV ...", flush=True)
        bundles, oof = ens_candidate(per_fold, Y_ens, diff, rounds, w)
        res = calibrate(
            f"ens(w={w})", attach_light_base(bundles), oof, masks, truth, policy
        )
        results.append(res)
        print(f"  weighted_cv={res['weighted']:.4f}")

    results.sort(key=lambda r: r["weighted"], reverse=True)
    print("\n===== 재대결 결과 (weighted CV, 그룹 fold, 동일 제약) =====")
    for res in results:
        print(f"{res['name']:16s} weighted_cv={res['weighted']:.4f}")
        for tier, cfg in res["tiers"].items():
            if cfg is None:
                print(f"    {tier:9s} feasible 없음")
            else:
                print(
                    f"    {tier:9s} β={cfg['beta']:.1f} margin={cfg['margin']:.2f} "
                    f"cv={cfg['cv_score']:.4f} "
                    f"max_used={cfg['max_used_15fold']:.3f}/{cfg['budget_multiplier']} "
                    f"stress_worst={cfg['stress_worst']:.3f}"
                )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "cv_seeds": list(CV_SEEDS),
            "folds": FOLDS,
            "group_prefix_chars": GROUP_PREFIX,
            "grouped_episodes": grouped,
            "safety_cap": SAFETY_CAP,
            "stress_cap": STRESS_CAP,
            "betas": list(BETAS),
            "margins": list(MARGINS),
            "ens_rounds": rounds if not args.smoke else None,
            "dev_used": False,
        },
        "results": results,
    }
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n보고서 저장: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
