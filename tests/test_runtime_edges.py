# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""런타임 엣지 입력 스모크 — 극단 입력에서도 유효 제출을 생성하는지.

빈 프롬프트는 프로토콜이 거부하므로 다루지 않는다.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from ossp_router import learned_router
from ossp_router.heuristic import main
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, parse_input

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _episode_batch(episodes):
    return parse_input(
        {
            "schema_version": 1,
            "challenge_id": "edge-test",
            "split": "synthetic",
            "episodes": episodes,
        }
    )


class RuntimeEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = learned_router.load_bundled_artifact()
        if cls.artifact is None:
            raise AssertionError(
                "번들 learned 아티팩트가 없습니다 — 포장 오류이므로 테스트를 "
                "skip하지 않고 실패로 처리합니다."
            )
        cls.policy = load_bundled_policy()

    def _assert_valid_submission(self, batch, tier="fast"):
        submission = learned_router.make_learned_submission(
            batch, self.policy, self.artifact, tier
        )
        decided = {d.episode_id for d in submission.decisions}
        self.assertEqual({ep.episode_id for ep in batch.episodes}, decided)
        for decision in submission.decisions:
            self.assertIn(decision.model_id, MODEL_IDS)
        return submission

    def test_single_episode_batch(self) -> None:
        batch = _episode_batch(
            [{"episode_id": "only-one", "prompt": "1 + 1은 얼마인가요?"}]
        )
        for tier in TIERS:
            self._assert_valid_submission(batch, tier)

    def test_very_long_prompt_150k_chars(self) -> None:
        long_text = ("Prove the identity carefully. " * 5000)[:150_000]
        batch = _episode_batch(
            [
                {"episode_id": "long-1", "prompt": long_text},
                {"episode_id": "short-1", "prompt": "짧은 질문입니다."},
            ]
        )
        self._assert_valid_submission(batch)

    def test_messages_form_and_mixed_unicode(self) -> None:
        batch = _episode_batch(
            [
                {
                    "episode_id": "messages-1",
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Traduis: bonjour, 你好, γειά"},
                    ],
                },
                {
                    "episode_id": "unicode-1",
                    "prompt": "수식 ∑ᵢ xᵢ² ≤ ∫f(x)dx 를 설명해줘. Émigré Straße 🌟",
                },
            ]
        )
        self._assert_valid_submission(batch)

    def test_main_end_to_end_single_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "out.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "challenge_id": "edge-test",
                        "split": "synthetic",
                        "episodes": [{"episode_id": "e2e-1", "prompt": "안녕하세요?"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rc = main(
                [
                    "--input",
                    str(input_path),
                    "--tier",
                    "premium",
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(0, rc)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(payload["decisions"]))


if __name__ == "__main__":
    unittest.main()
