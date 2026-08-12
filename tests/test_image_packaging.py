# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""컨테이너 포장 회귀 방지 검사.

v1.2에서 .dockerignore allowlist에 ens_router.py와 ens-artifact.v1.json이
빠져 컨테이너가 ImportError로 죽는 사고가 있었다 (전 tier 실행 실패 = 0점).
이 테스트는 런타임 진입점에서 도달 가능한 import 폐쇄와 참조 리소스가
전부 allowlist에 올라 있는지 정적으로 검증해 재발을 막는다.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ossp_router"
ENTRY_MODULE = "heuristic"  # container/entrypoint.py가 import하는 모듈

_FROM_RELATIVE = re.compile(r"^\s*from \.(\w+) import", re.MULTILINE)
_FROM_RELATIVE_BARE = re.compile(r"^\s*from \. import ([\w ,]+)", re.MULTILINE)
_FROM_ABSOLUTE = re.compile(r"^\s*from ossp_router\.(\w+) import", re.MULTILINE)
_FROM_ABSOLUTE_BARE = re.compile(r"^\s*from ossp_router import ([\w ,]+)", re.MULTILINE)
_IMPORT_ABSOLUTE = re.compile(r"^\s*import ossp_router\.(\w+)", re.MULTILINE)
_JSON_LITERAL = re.compile(r"[\w][\w.-]*\.json")


def _allowlisted_files() -> set[str]:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return {
        line[1:] for line in lines if line.startswith("!") and not line.endswith("/")
    }


def _package_imports(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in (_FROM_RELATIVE, _FROM_ABSOLUTE, _IMPORT_ABSOLUTE):
        found.update(pattern.findall(text))
    for pattern in (_FROM_RELATIVE_BARE, _FROM_ABSOLUTE_BARE):
        for group in pattern.findall(text):
            found.update(name.strip() for name in group.split(","))
    return {name for name in found if (SRC / f"{name}.py").is_file()}


def _runtime_closure() -> tuple[set[str], set[str]]:
    """진입 모듈에서 도달 가능한 (모듈 집합, 참조 리소스 파일 집합)."""
    seen: set[str] = set()
    resources: set[str] = set()
    queue = [ENTRY_MODULE]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        text = (SRC / f"{module}.py").read_text(encoding="utf-8")
        queue.extend(_package_imports(text) - seen)
        for name in _JSON_LITERAL.findall(text):
            if (SRC / "resources" / name).is_file():
                resources.add(name)
    return seen, resources


class ImagePackagingTest(unittest.TestCase):
    def test_runtime_import_closure_is_allowlisted(self) -> None:
        allowlisted = _allowlisted_files()
        modules, resources = _runtime_closure()
        missing = []
        for module in sorted(modules | {"__init__"}):
            path = f"src/ossp_router/{module}.py"
            if path not in allowlisted:
                missing.append(path)
        for name in sorted(resources):
            path = f"src/ossp_router/resources/{name}"
            if path not in allowlisted:
                missing.append(path)
        for path in (
            "src/ossp_router/resources/__init__.py",
            "container/Dockerfile",
            "container/entrypoint.py",
        ):
            if path not in allowlisted:
                missing.append(path)
        self.assertEqual(
            [],
            missing,
            f".dockerignore allowlist에 런타임 필수 파일이 빠졌습니다: {missing}",
        )

    def test_allowlisted_files_exist(self) -> None:
        stale = [
            path for path in sorted(_allowlisted_files()) if not (ROOT / path).is_file()
        ]
        self.assertEqual(
            [], stale, f".dockerignore allowlist에 존재하지 않는 파일: {stale}"
        )

    def test_closure_reaches_expected_runtime_modules(self) -> None:
        """폐쇄 계산 자체의 자기 검증 — 핵심 모듈이 실제로 포착되는지."""
        modules, resources = _runtime_closure()
        self.assertIn("learned_router", modules)
        self.assertIn("learned_features", modules)
        self.assertIn("learned_policy", modules)
        self.assertIn("protocol", modules)
        self.assertIn("learned-router.v1.json", resources)
        self.assertIn("routing-policy.v1.json", resources)


if __name__ == "__main__":
    unittest.main()
