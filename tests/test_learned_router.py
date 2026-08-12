# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from ossp_router import learned_router
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_bundled_policy, parse_input

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _batch(episode_ids=("first", "second", "third", "fourth")):
    return parse_input(
        {
            "schema_version": 1,
            "challenge_id": "learned-test",
            "split": "synthetic",
            "episodes": [
                {
                    "episode_id": episode_ids[0],
                    "prompt": "오늘 날씨를 짧게 설명해 주세요.",
                },
                {
                    "episode_id": episode_ids[1],
                    "prompt": (
                        "Prove x^2 + 2*x + 1 = (x+1)^2 and derive every step. "
                        "Numbers: 12, 24, 48, 96."
                    ),
                },
                {
                    "episode_id": episode_ids[2],
                    "messages": [
                        {"role": "system", "content": "Write valid Python."},
                        {
                            "role": "user",
                            "content": "```python\ndef solve(values):\n    return values\n```",
                        },
                        {"role": "assistant", "content": "Explain complexity."},
                    ],
                },
                {
                    "episode_id": episode_ids[3],
                    "prompt": "다음 문장을 영어로 번역해 주세요: 좋은 아침입니다.",
                },
            ],
        }
    )


def _reorder(batch):
    return parse_input(
        {
            "schema_version": batch.schema_version,
            "challenge_id": batch.challenge_id,
            "split": batch.split,
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    **(
                        {"prompt": episode.prompt}
                        if episode.prompt is not None
                        else {
                            "messages": [
                                {"role": message.role, "content": message.content}
                                for message in episode.messages or ()
                            ]
                        }
                    ),
                }
                for episode in reversed(batch.episodes)
            ],
        }
    )


def _content_decisions(batch, submission):
    model_by_id = {
        decision.episode_id: decision.model_id for decision in submission.decisions
    }
    return {
        episode_text(episode): model_by_id[episode.episode_id]
        for episode in batch.episodes
    }


class LearnedRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_bundled_policy()
        self.artifact = learned_router.load_bundled_artifact()
        # 아티팩트 부재는 포장 오류이므로 skip이 아니라 실패로 처리한다.
        self.assertIsNotNone(
            self.artifact,
            "동봉된 learned 아티팩트가 없습니다 (analysis/train_linear.py 실행 필요)",
        )

    def test_artifact_matches_bundled_policy(self) -> None:
        submission = learned_router.make_learned_submission(
            _batch(), self.policy, self.artifact, "balanced"
        )
        self.assertEqual(4, len(submission.decisions))

    def test_order_does_not_change_content_decisions(self) -> None:
        original = _batch()
        reordered = _reorder(original)
        for tier in ("fast", "balanced", "premium"):
            with self.subTest(tier=tier):
                self.assertEqual(
                    _content_decisions(
                        original,
                        learned_router.make_learned_submission(
                            original, self.policy, self.artifact, tier
                        ),
                    ),
                    _content_decisions(
                        reordered,
                        learned_router.make_learned_submission(
                            reordered, self.policy, self.artifact, tier
                        ),
                    ),
                )

    def test_changed_ids_do_not_change_content_decisions(self) -> None:
        original = _batch()
        changed = _batch(("opaque-a", "opaque-b", "opaque-c", "opaque-d"))
        for tier in ("fast", "balanced", "premium"):
            with self.subTest(tier=tier):
                self.assertEqual(
                    _content_decisions(
                        original,
                        learned_router.make_learned_submission(
                            original, self.policy, self.artifact, tier
                        ),
                    ),
                    _content_decisions(
                        changed,
                        learned_router.make_learned_submission(
                            changed, self.policy, self.artifact, tier
                        ),
                    ),
                )

    def test_repeated_router_run_is_byte_deterministic(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary)
            outputs = []
            for index in range(2):
                output = target / f"submission-{index}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ossp_router.heuristic",
                        "--input",
                        str(ROOT / "data/toy/inputs.json"),
                        "--tier",
                        "balanced",
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                outputs.append(output.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual("balanced", json.loads(outputs[0])["tier"])

    def test_predictions_are_content_only_and_finite(self) -> None:
        batch = _batch()
        prediction = learned_router.predict_episode(batch.episodes[0], self.artifact)
        self.assertEqual(set(prediction), {"ax31-light", "ax31", "axk1-think"})
        for score, cost in prediction.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertGreater(cost, 0.0)

    def test_artifact_validation_rejects_truncated_weights(self) -> None:
        payload = json.loads(
            (ROOT / "src/ossp_router/resources/learned-router.v1.json").read_text(
                encoding="utf-8"
            )
        )
        payload["weights"]["score"]["ax31"] = payload["weights"]["score"]["ax31"][:-1]
        from ossp_router.protocol import ProtocolError

        with self.assertRaises(ProtocolError):
            learned_router.parse_artifact(payload)


if __name__ == "__main__":
    unittest.main()
