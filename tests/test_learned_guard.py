# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""컨텍스트 소진 가드(k_guard) — 런타임·학습측 동등성과 발화 조건."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

from ossp_router.learned_features import k_guard
from ossp_router.protocol import load_input

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_INPUT = ROOT / "data" / "materialized" / "train" / "inputs.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


os2_features = _load_module("os2_features_guard", ROOT / "analysis/os2_features.py")

FIRES = (
    "Is 118034857 prime?",
    "What are the prime factors of 2567663?",
    "List the prime factors of 38948129.",
    "Determine b so that -b**3 - 2262108*b**2 - 275965817*b - 7973500650 = 0.",
    "What is j in -j**2/5 + 34852635524*j/5 - 303676550742196688644/5 = 0?",
    "Solve 12345678*x^2 - 3*x + 1 = 0",
    "Suppose -20*x**5 + 119247535*x**4 + 626049925*x**3 + 1103040550*x**2 = 0.\nFind x.",
    # DeepMind Mathematics 표준 문형 — `= 0` 뒤에 같은 줄에서 문장이 이어진다
    "Suppose -20*x**5 + 119247535*x**4 + 626049925*x**3 + 715485820*x + 1192475 = 0. What is x?",
    "Solve -j**2 + 34852635524*j - 303676550742196688644 = 0 for j.",
    "Let 3*b**3 - 2262108*b**2 + 7973500650 = 0. Determine b.",
)
SILENT = (
    "Solve -3*x**2 + 5*x - 2 = 0.",  # 큰 정수가 없다
    "Solve -3*x**2 + 123456*x - 2 = 0.",  # 6자리 — 자릿수 하한(7) 경계
    "Is 97 prime?",
    "Is 999983 prime?",  # 6자리 소수 — 자릿수 하한 경계
    "Solve 12345678*x**1 + 3 = 0",  # 1차 — 차수 하한(2) 경계
    "Solve 12345678*x^1 + 3 = 0",
    "Let n = 12345678. Solve 2*x + n = 0.",  # 1차식 — 거듭제곱이 없다
    "Calculate 12345678 * 3.",
    "def f(students):\n    seatlist = students\n    seatlist.reverse()\n    return seatlist\n\nassert f(['r', '9']) == ??",
    "Compute 2**10 + 12345678 and explain.",  # `= 0` 방정식이 아니다
    "3.1415926*x**2 - 1 = 0.5",  # `= 0.5`는 0이 아니다
    "MOD = 1000000007\nans = 0\nfor i in range(n):\n    ans = (ans + i ** 2) % MOD",  # 거듭제곱과 `= 0`이 다른 줄
    "int a = 12345678;\nint b = a ^ 2;\nint c = 0;",  # XOR — 같은 줄에 `= 0`이 없다
    "Item **2**\nID 1234567\nbalance = 0",  # 마크다운 볼드
    "",
)


class KGuardTest(unittest.TestCase):
    def test_fires_on_context_exhaustion_prompt_shapes(self) -> None:
        for text in FIRES:
            self.assertTrue(k_guard(text), text)

    def test_silent_on_ordinary_prompts(self) -> None:
        for text in SILENT:
            self.assertFalse(k_guard(text), text)

    def test_reads_only_a_bounded_prefix(self) -> None:
        padding = "a" * 6000
        self.assertFalse(k_guard(padding + "\nIs 118034857 prime?"))
        self.assertTrue(k_guard("Is 118034857 prime?\n" + padding))

    def test_training_copy_agrees_on_synthetic(self) -> None:
        for text in FIRES + SILENT:
            self.assertEqual(os2_features.k_guard(text), k_guard(text), text)

    @unittest.skipUnless(
        TRAIN_INPUT.is_file(),
        "materialized 공개 train이 없습니다 (로컬 생성 데이터라 저장소 미포함)",
    )
    def test_training_copy_agrees_on_full_public_train(self) -> None:
        from ossp_router.heuristic import episode_text

        inputs = load_input(TRAIN_INPUT)
        fired = 0
        for episode in inputs.episodes:
            text = episode_text(episode)
            self.assertEqual(os2_features.k_guard(text), k_guard(text), episode.episode_id)
            fired += k_guard(text)
        # 가드는 좁아야 한다 — 공개 train에서 1% 미만.
        self.assertLess(fired / len(inputs.episodes), 0.01)


if __name__ == "__main__":
    unittest.main()
