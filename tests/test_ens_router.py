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

from ossp_router import ens_router
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_bundled_policy, parse_input

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _batch(episode_ids=("first", "second", "third", "fourth")):
    return parse_input(
        {
            "schema_version": 1,
            "challenge_id": "ens-test",
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


class EnsRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_bundled_policy()
        self.artifact = ens_router.load_bundled_artifact()
        if self.artifact is None:
            self.skipTest(
                "동봉된 ens 아티팩트가 없습니다 (analysis/train_ens.py 실행 필요)"
            )

    def test_artifact_matches_bundled_policy(self) -> None:
        submission = ens_router.make_ens_submission(
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
                        ens_router.make_ens_submission(
                            original, self.policy, self.artifact, tier
                        ),
                    ),
                    _content_decisions(
                        reordered,
                        ens_router.make_ens_submission(
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
                        ens_router.make_ens_submission(
                            original, self.policy, self.artifact, tier
                        ),
                    ),
                    _content_decisions(
                        changed,
                        ens_router.make_ens_submission(
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

    def test_predictions_are_content_only(self) -> None:
        batch = _batch()
        scores, costs = ens_router.predict_episode(batch.episodes[0], self.artifact)
        for mapping in (scores, costs):
            self.assertEqual(set(mapping), {"ax31-light", "ax31", "axk1-think"})
        self.assertLessEqual(costs["ax31-light"], costs["ax31"])
        self.assertLessEqual(costs["ax31"], costs["axk1-think"])
        for value in scores.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
