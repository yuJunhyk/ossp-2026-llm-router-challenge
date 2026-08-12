# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Budget-aware greedy allocation for the learned router (pure stdlib).

Multi-choice-knapsack greedy over predicted (score, cost) per model. The
offline calibration (work/train_router.py) and the container runtime share
this exact module.

analysis/os2_policy.py(예측기 재대결·캘리브레이션에 쓴 학습측 사본)와
문자 그대로 동일해야 한다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

MODELS = ("ax31-light", "ax31", "axk1-think")


def allocate(
    predictions: List[Dict[str, Tuple[float, float]]],
    budget_multiplier: float,
    margin: float,
) -> List[str]:
    """predictions[i] = {model: (score, cost)}. Returns per-episode model ids.

    Starts from all-light and greedily upgrades along each episode's convex
    hull until the estimated budget (light-baseline x multiplier x margin)
    is exhausted.
    """
    n = len(predictions)
    choice = ["ax31-light"] * n
    est_base = sum(p["ax31-light"][1] for p in predictions)
    if est_base <= 0:
        return choice
    budget = est_base * budget_multiplier * margin
    spent = est_base

    increments = []  # (ratio, episode, step_index, target_model, d_score, d_cost)
    for i, p in enumerate(predictions):
        s_l, c_l = p["ax31-light"]
        s_a, c_a = p["ax31"]
        s_k, c_k = p["axk1-think"]
        # enforce cost monotonicity against prediction inversions
        c_l = max(c_l, 1e-12)
        c_a = max(c_a, c_l * 1.01)
        c_k = max(c_k, c_a * 1.01)
        steps = []
        if s_a > s_l and s_k > s_a:
            r1 = (s_a - s_l) / (c_a - c_l)
            r2 = (s_k - s_a) / (c_k - c_a)
            if r2 >= r1:  # concave: merge into a single light->think step
                steps = [("axk1-think", s_k - s_l, c_k - c_l)]
            else:
                steps = [
                    ("ax31", s_a - s_l, c_a - c_l),
                    ("axk1-think", s_k - s_a, c_k - c_a),
                ]
        elif s_a > s_l:
            steps = [("ax31", s_a - s_l, c_a - c_l)]
        elif s_k > s_l:
            steps = [("axk1-think", s_k - s_l, c_k - c_l)]
        # 동률(ratio가 정확히 같은 증분)의 순서를 문항 위치가 아니라 예측값
        # 튜플(내용에서만 유도됨)로 깨서, 입력 순서를 뒤섞어도 같은 내용이면
        # 같은 배분이 나오게 한다 (ID·순서 의존성 점검 대응).
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
            continue  # earlier step skipped for budget: skip followers too
        if spent + dc > budget:
            continue
        spent += dc
        choice[i] = target
        taken_step[i] = step_index
    return choice
