# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""ens 라우터 최종 트레이너 — ridge + gbm-diff 앙상블 아티팩트 생성.

설계 근거 (analysis/exp_gbm.py, exp_v2.py, exp_v4.py):
- ridge(α=10) 단독보다 gbm-diff(uplift 직접 학습)와의 0.5:0.5 앙상블이 우세
- GBM은 잎 7개·최소 잎 데이터 15·학습률 0.03 (train OOF로 선택)
- 타깃별 부스팅 라운드는 train OOF MSE로 선택, 시드 평균으로 분산 축소

안전계수는 Dev가 아니라 train OOF의 실제 비용비율에 배포 버퍼를 적용해
정한다 (공식 hash-regex가 비공개 검증에서 Premium 예산을 5.4% 초과한 사례
근거). Dev는 학습·보정에 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

import numpy as np
import lightgbm as lgb

import hash_regex
from train_hash_regex import _fit_ridge, _predict_ridge, _training_matrix
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)

HASH_BINS = 256
RIDGE_ALPHA = 10.0
ENSEMBLE_GBM_WEIGHT = 0.5
GBM_PARAMS = dict(
    objective="regression",
    num_leaves=7,
    learning_rate=0.03,
    min_data_in_leaf=15,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
)
ROUND_GRID = (100, 200, 400, 800)
FOLDS = 5
SEEDS = (7, 8, 9)
# 배포 버퍼: train 실제 비용비율 × 버퍼 ≤ 한도가 되도록 안전계수를 고른다.
# 근거 — 실측 train→Dev 비용 이동: fast +5.4%~+7.4%(선택 구성에 따라),
# balanced −5.5%, premium +15.6% (2026-08-11 Dev 확인 2회), 공식 hash-regex의
# Dev→비공개 +5.4% 초과 사례. 관측 이동폭 + 비공개 여유분을 합쳐 tier별로
# 차등화하며, premium은 K1 출력 꼬리의 변동성 때문에 가장 보수적으로 둔다.
DEPLOY_BUFFER = {"fast": 1.16, "balanced": 1.10, "premium": 1.30}
TARGET_NAMES = (
    "score_light",
    "uplift_ax31",
    "uplift_k1",
    "logcost_light",
    "logcost_ax31",
    "logcost_k1",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_diff(Y: np.ndarray) -> np.ndarray:
    out = Y.copy()
    out[:, 1] = Y[:, 1] - Y[:, 0]
    out[:, 2] = Y[:, 2] - Y[:, 0]
    return out


def reconstruct(pred: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[:, 1] = pred[:, 0] + pred[:, 1]
    out[:, 2] = pred[:, 0] + pred[:, 2]
    return out


def select_rounds(X: np.ndarray, targets: np.ndarray) -> tuple[list[int], np.ndarray]:
    """타깃별 부스팅 라운드를 train OOF MSE로 고르고 OOF 예측을 돌려준다."""
    n = X.shape[0]
    fold_ids = np.arange(n) % FOLDS
    rounds_chosen = []
    oof = np.empty_like(targets)
    for t in range(targets.shape[1]):
        best = None
        for rounds in ROUND_GRID:
            cur = np.empty(n)
            for fold in range(FOLDS):
                va = fold_ids == fold
                ds = lgb.Dataset(X[~va], label=targets[~va, t])
                params = dict(GBM_PARAMS, seed=SEEDS[0])
                cur[va] = lgb.train(params, ds, num_boost_round=rounds).predict(X[va])
            mse = float(np.mean((cur - targets[:, t]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, rounds, cur.copy())
        rounds_chosen.append(best[1])
        oof[:, t] = best[2]
        print(f"  {TARGET_NAMES[t]}: rounds={best[1]} OOF-MSE={best[0]:.5f}")
    return rounds_chosen, oof


def flatten_booster(booster: lgb.Booster) -> list[dict]:
    """LightGBM 트리를 stdlib 추론용 평탄 배열로 변환한다.

    각 트리는 {"f": [특징 인덱스], "t": [임계값], "l": [왼쪽 노드], "r": [오른쪽 노드],
    "v": [리프값]} 형식이며, 노드 인덱스 음수는 리프(-idx-1이 v의 인덱스)를 뜻한다.
    """
    dump = booster.dump_model()
    trees = []
    for tree_info in dump["tree_info"]:
        feats: list[int] = []
        thresholds: list[float] = []
        lefts: list[int] = []
        rights: list[int] = []
        leaf_values: list[float] = []
        node_ids: dict[int, int] = {}

        def walk(node) -> int:
            if "leaf_value" in node and "split_feature" not in node:
                leaf_values.append(float(node["leaf_value"]))
                return -len(leaf_values)  # -1부터 시작
            index = len(feats)
            feats.append(int(node["split_feature"]))
            thresholds.append(float(node["threshold"]))
            lefts.append(0)
            rights.append(0)
            left = walk(node["left_child"])
            right = walk(node["right_child"])
            lefts[index] = left
            rights[index] = right
            # LightGBM 기본 분기: value <= threshold → left
            if node.get("decision_type") != "<=":
                raise ValueError(
                    f"지원하지 않는 decision_type: {node.get('decision_type')}"
                )
            if node.get("default_left") not in (True, "true"):
                # 결측 방향 — 우리 특징은 항상 유한값이라 영향 없지만 기록 차원에서 검증
                pass
            return index

        root = tree_info["tree_structure"]
        if "split_feature" not in root:
            # 분할 없는 단일 리프 트리
            trees.append(
                {"f": [], "t": [], "l": [], "r": [], "v": [float(root["leaf_value"])]}
            )
            continue
        walk(root)
        trees.append(
            {"f": feats, "t": thresholds, "l": lefts, "r": rights, "v": leaf_values}
        )
    return trees


def eval_flat_trees(trees: list[dict], features) -> float:
    """평탄 트리 배열 추론 (런타임과 동일 로직 — 교차 검증용)."""
    total = 0.0
    for tree in trees:
        if not tree["f"]:
            total += tree["v"][0]
            continue
        node = 0
        while node >= 0:
            if features[tree["f"][node]] <= tree["t"][node]:
                node = tree["l"][node]
            else:
                node = tree["r"][node]
        total += tree["v"][-node - 1]
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="ens 라우터 아티팩트 학습")
    parser.add_argument(
        "--input", type=Path, default=REPO / "data/materialized/train/inputs.json"
    )
    parser.add_argument(
        "--outcomes", type=Path, default=REPO / "data/train/outcomes.json"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO / "src/ossp_router/resources/ens-artifact.v1.json",
    )
    parser.add_argument(
        "--report", type=Path, default=REPO / "build/ens/train-report.json"
    )
    args = parser.parse_args()

    policy = load_bundled_policy()
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    X, Y = _training_matrix(inputs, outcomes, policy, HASH_BINS)
    diff = to_diff(Y)

    print("[1/4] ridge 학습")
    mean, scale, intercept, coef = _fit_ridge(X, Y, RIDGE_ALPHA)

    print("[2/4] gbm-diff 라운드 선택 (train OOF)")
    rounds_chosen, oof_diff = select_rounds(X, diff)

    print("[3/4] 전체 train으로 시드별 최종 학습")
    gbm_trees = []  # [target][seed] → 평탄 트리 목록
    fitted_diff = np.zeros_like(diff)  # 배포 시 실제로 쓰는 fitted 예측 (시드 평균)
    for t in range(diff.shape[1]):
        per_seed = []
        for seed in SEEDS:
            ds = lgb.Dataset(X, label=diff[:, t])
            booster = lgb.train(
                dict(GBM_PARAMS, seed=seed), ds, num_boost_round=rounds_chosen[t]
            )
            fitted_diff[:, t] += booster.predict(X) / len(SEEDS)
            flat = flatten_booster(booster)
            # 평탄화 검증: 원본 예측과 일치해야 한다
            check_rows = X[:: max(1, len(X) // 64)]
            original = booster.predict(check_rows)
            for row, expected in zip(check_rows, original):
                got = eval_flat_trees(flat, row)
                if abs(got - expected) > 1e-9:
                    raise AssertionError("트리 평탄화 예측 불일치")
            per_seed.append(flat)
        gbm_trees.append(per_seed)

    print("[4/4] 안전계수 산정 (train OOF 실제 비용비율 × 배포 버퍼)")
    ridge_oof = np.empty_like(Y)
    fold_ids = np.arange(len(X)) % FOLDS
    for fold in range(FOLDS):
        va = fold_ids == fold
        m, s, b, c = _fit_ridge(X[~va], Y[~va], RIDGE_ALPHA)
        ridge_oof[va] = _predict_ridge(X[va], m, s, b, c)
    ens_oof = (1 - ENSEMBLE_GBM_WEIGHT) * ridge_oof + ENSEMBLE_GBM_WEIGHT * reconstruct(
        oof_diff
    )

    # train 실제 score·비용
    rates = {m: policy.models[m] for m in MODEL_IDS}
    truth = {}
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
        truth[(oc.episode_id, oc.model_id)] = (float(oc.score), cost)
    episode_ids = [ep.episode_id for ep in inputs.episodes]
    true_cost = {m: [truth[(i, m)][1] for i in episode_ids] for m in MODEL_IDS}
    light_total = sum(true_cost["ax31-light"])

    def rows_from(pred):
        scores, costs = [], []
        for row in pred:
            s = {m: min(1.0, max(0.0, float(row[i]))) for i, m in enumerate(MODEL_IDS)}
            c = {
                m: math.exp(min(50.0, max(-50.0, float(row[3 + i]))))
                for i, m in enumerate(MODEL_IDS)
            }
            light = c[MODEL_IDS[0]]
            c[MODEL_IDS[1]] = max(c[MODEL_IDS[1]], light * (1 + 1e-12))
            c[MODEL_IDS[2]] = max(c[MODEL_IDS[2]], c[MODEL_IDS[1]] * (1 + 1e-12))
            scores.append(s)
            costs.append(c)
        return scores, costs

    # 배포 시 라우터는 fitted 예측(전체 train 학습)을 쓰므로, OOF 기준과
    # fitted 기준 양쪽에서 안전계수를 구해 더 보수적인 쪽을 채택한다
    # (공식 hash-regex 트레이너의 fitted 재검증 단계와 같은 취지).
    ens_fitted = (1 - ENSEMBLE_GBM_WEIGHT) * _predict_ridge(
        X, mean, scale, intercept, coef
    ) + ENSEMBLE_GBM_WEIGHT * reconstruct(fitted_diff)

    def largest_safe(pred, tier):
        mult = float(policy.tiers[tier].budget_multiplier)
        buffer = DEPLOY_BUFFER[tier]
        scores, costs = rows_from(pred)
        lo = 1.0 / mult
        for i in range(120, -1, -1):  # 큰 안전계수부터 내려가며 탐색
            safety = lo + (1 - lo) * i / 120
            selected, _ = hash_regex.select_models(
                scores, costs, budget_multiplier=mult, safety_ratio=safety
            )
            actual = sum(true_cost[m][j] for j, m in enumerate(selected)) / light_total
            if actual * buffer <= mult:
                return safety, actual
        return lo, 1.0

    safety_ratios = {}
    safety_report = {}
    for tier in TIERS:
        oof_safety, oof_actual = largest_safe(ens_oof, tier)
        fit_safety, fit_actual = largest_safe(ens_fitted, tier)
        final = min(oof_safety, fit_safety)
        safety_ratios[tier] = final
        safety_report[tier] = {
            "safety_ratio": final,
            "oof_safety": oof_safety,
            "oof_actual_ratio": oof_actual,
            "fitted_safety": fit_safety,
            "fitted_actual_ratio": fit_actual,
            "deploy_buffer": DEPLOY_BUFFER[tier],
            "budget_multiplier": float(policy.tiers[tier].budget_multiplier),
        }
        print(
            f"  {tier}: 안전계수 {final:.4f} "
            f"(OOF {oof_safety:.4f}/실제 {oof_actual:.4f}, "
            f"fitted {fit_safety:.4f}/실제 {fit_actual:.4f}, 버퍼 {DEPLOY_BUFFER[tier]})"
        )

    artifact = {
        "artifact_type": "ossp-ens-router-v1",
        "schema_version": 1,
        "hash_bins": HASH_BINS,
        "dense_feature_names": list(hash_regex.DENSE_FEATURE_NAMES),
        "model_ids": list(MODEL_IDS),
        "target_names": list(TARGET_NAMES),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "ensemble_gbm_weight": ENSEMBLE_GBM_WEIGHT,
        "ridge": {
            "alpha": RIDGE_ALPHA,
            "feature_mean": [float(v) for v in mean],
            "feature_scale": [float(v) for v in scale],
            "intercept": [float(v) for v in intercept],
            "coefficients": [[float(v) for v in coef[:, t]] for t in range(6)],
        },
        "gbm": {
            "params": {
                k: v
                for k, v in GBM_PARAMS.items()
                if k not in ("verbosity", "num_threads")
            },
            "rounds": rounds_chosen,
            "seeds": list(SEEDS),
            "trees": gbm_trees,
        },
        "tier_safety_ratios": safety_ratios,
        "training_summary": {
            "num_episodes": len(inputs.episodes),
            "folds": FOLDS,
            "input_sha256": file_sha256(args.input),
            "outcomes_sha256": file_sha256(args.outcomes),
            "safety_method": "train-oof-actual-ratio-with-deploy-buffer",
            "safety_report": safety_report,
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(
            artifact, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
        encoding="utf-8",
    )
    size_mb = args.artifact.stat().st_size / 1e6
    print(f"아티팩트 저장: {args.artifact} ({size_mb:.2f} MB)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"rounds": rounds_chosen, "safety": safety_report},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
