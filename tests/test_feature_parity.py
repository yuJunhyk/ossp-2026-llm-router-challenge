# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""학습(baselines/hash_regex.py)–런타임(ens_router) 특징 추출 동등성.

아티팩트는 학습측 특징으로 만들어지고 추론은 런타임 재구현으로 돌므로,
두 구현이 비트 단위로 같지 않으면 조용한 오예측이 생긴다. 합성 프롬프트는
항상 검사하고, materialized 공개 train이 있으면 전수(1,760문항)도 검사한다.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

from ossp_router import ens_router
from ossp_router.protocol import load_input, parse_input

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_INPUT = ROOT / "data" / "materialized" / "train" / "inputs.json"
HASH_BINS = 256


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hash_regex = _load_module("hash_regex", ROOT / "baselines/hash_regex.py")

SYNTHETIC_TEXTS = (
    "오늘 날씨를 짧게 설명해 주세요.",
    "Prove x^2 + 2*x + 1 = (x+1)^2 step by step. Numbers: 12, 24, 48.",
    "```python\ndef solve(values):\n    return sorted(values)\n```",
    "수식 ∑ᵢ xᵢ² ≤ ∫f(x)dx 를 설명해줘. Émigré Straße 🌟",
    "A. 첫째 B. 둘째 C. 셋째 D. 넷째 중 고르시오.\n" * 20,
    ("Prove the identity carefully. " * 5000)[:150_000],
    "?",
)


def _synthetic_episodes():
    return parse_input(
        {
            "schema_version": 1,
            "challenge_id": "parity-test",
            "split": "synthetic",
            "episodes": [
                {"episode_id": f"parity-{i}", "prompt": text}
                for i, text in enumerate(SYNTHETIC_TEXTS)
            ],
        }
    ).episodes


os2_features = _load_module("os2_features", ROOT / "analysis/os2_features.py")


class LearnedFeatureParityTest(unittest.TestCase):
    """학습측(analysis/os2_features)–런타임(learned_features) 동등성."""

    def test_synthetic_prompts_bit_equal(self) -> None:
        from ossp_router.learned_features import extract_sparse

        for text in SYNTHETIC_TEXTS:
            self.assertEqual(os2_features.extract_sparse(text), extract_sparse(text))

    @unittest.skipUnless(
        TRAIN_INPUT.is_file(),
        "materialized 공개 train이 없습니다 (로컬 생성 데이터라 저장소 미포함)",
    )
    def test_full_public_train_bit_equal(self) -> None:
        from ossp_router.heuristic import episode_text
        from ossp_router.learned_features import extract_sparse

        inputs = load_input(TRAIN_INPUT)
        mismatch = 0
        for episode in inputs.episodes:
            text = episode_text(episode)
            if os2_features.extract_sparse(text) != extract_sparse(text):
                mismatch += 1
        self.assertEqual(0, mismatch)


class FeatureParityTest(unittest.TestCase):
    def test_synthetic_prompts_bit_equal(self) -> None:
        for episode in _synthetic_episodes():
            expected = hash_regex.raw_feature_vector(episode, HASH_BINS)
            got = ens_router.raw_feature_vector(episode, HASH_BINS)
            self.assertEqual(list(expected), list(got))

    @unittest.skipUnless(
        TRAIN_INPUT.is_file(),
        "materialized 공개 train이 없습니다 (로컬 생성 데이터라 저장소 미포함)",
    )
    def test_full_public_train_bit_equal(self) -> None:
        inputs = load_input(TRAIN_INPUT)
        mismatch = 0
        for episode in inputs.episodes:
            if list(hash_regex.raw_feature_vector(episode, HASH_BINS)) != list(
                ens_router.raw_feature_vector(episode, HASH_BINS)
            ):
                mismatch += 1
        self.assertEqual(0, mismatch)


if __name__ == "__main__":
    unittest.main()
