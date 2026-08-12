# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-content-only feature extraction for the learned router.

Pure standard library, deterministic (crc32 hashing). Shared verbatim by the
offline training pipeline and the container runtime. Metadata such as
``episode_id`` is never an input.

출처: 동일 참가자의 독립 재구현 세션(learned_features.py)에서 예측기
재대결(analysis/rematch.py)을 위해 그대로 이식했다.
"""

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from typing import Dict, Tuple

HASH_DIM = 8192  # 2**13
NUMERIC_DIM = 40
TOTAL_DIM = NUMERIC_DIM + HASH_DIM

_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[가-힣]+|[一-鿿]+")
_MATH_RE = re.compile(
    r"\\(?:frac|sqrt|sum|int|prod|lim|log|sin|cos|tan|binom|cdot|times|leq|geq|neq|pi|alpha|beta|theta|lambda|mathbb|mathrm|begin|end)"
    r"|[=+\-*/^<>]{1,}|\$\$?|∑|∫|√|×|÷|≤|≥|≠"
)
_CODE_RE = re.compile(
    r"```|\bdef\b|\bclass\b|\bimport\b|\breturn\b|\bfunction\b|\bconst\b|\bvar\b|"
    r"\bpublic\b|\bprintf\b|#include|SELECT\s+.+\s+FROM|[{};]"
)
_REASON_RE = re.compile(
    r"\bprove\b|\btheorem\b|\blemma\b|\bstep[- ]by[- ]step\b|\bexplain why\b|"
    r"\bderive\b|\bshow that\b|\bfind all\b|\bhow many\b|\bremainder\b|\bmodulo\b|"
    r"증명|풀이|단계별|이유를|왜|추론|과정을",
    re.IGNORECASE,
)
_SIMPLE_RE = re.compile(
    r"\bsummari[sz]e\b|\btranslate\b|\brewrite\b|\bparaphrase\b|\blist\b|"
    r"\bwhat is\b|\bdefine\b|\bround\b|\bconvert\b|"
    r"요약|번역|나열|정의|반올림|변환",
    re.IGNORECASE,
)
_AIME_RE = re.compile(
    r"\btriangle\b|\bpolynomial\b|\binteger[s]?\b|\bdigits?\b|\bdivisible\b|"
    r"\bsequence\b|\bprobability\b|\bgeometry\b|\bcircle\b|\broots?\b|\bprime\b",
    re.IGNORECASE,
)
_MC_OPTION_RE = re.compile(r"(?:^|\n|\s)[A-E]\.\s")
_MC_HEAD_RE = re.compile(r"^\s*Question\s*:", re.IGNORECASE)
_RULE_CHAIN_RE = re.compile(
    r"\b(?:does not|do not)\b|\bIf some(?:one|thing)\b|\bthen the\b|\bis true\b",
)
_CODE_EXEC_RE = re.compile(r"assert\s+\w+\s*\(|==\s*\?\?|\bdef f\(")
_DM_MATH_RE = re.compile(
    r"^\s*(?:Let\s|Suppose\s|What is|Calculate|Simplify|Solve|Round|Find|"
    r"Evaluate|Factor|Expand|Differentiate|Divide|Multiply|Sort|Convert|"
    r"Is\s|How many|Put|List|Which is|What comes)",
)
_WORD_PROBLEM_RE = re.compile(
    r"\bhow (?:many|much|old|long|far)\b.*\?|\bin total\b|\baltogether\b|"
    r"\beach\b.*\bcost\b|\bper (?:day|hour|week|month)\b",
    re.IGNORECASE | re.DOTALL,
)


def _bucket(token: str, salt: str) -> Tuple[int, float]:
    h = zlib.crc32((salt + "\x1f" + token).encode("utf-8"))
    index = h % HASH_DIM
    sign = 1.0 if (h >> 17) & 1 else -1.0
    return index, sign


def extract_sparse(text: str) -> Dict[int, float]:
    """Sparse feature vector {index: value}; indexes < NUMERIC_DIM are numeric."""
    features: Dict[int, float] = {}

    n_chars = len(text)
    clipped = text[:20000]
    words = _WORD_RE.findall(clipped)
    n_words = len(words)
    lines = clipped.splitlines() or [""]

    hangul = sum(1 for c in clipped if "가" <= c <= "힣")
    ascii_alpha = sum(1 for c in clipped if c.isascii() and c.isalpha())
    digits = sum(1 for c in clipped if c.isdigit())
    spaces = sum(1 for c in clipped if c.isspace())
    puncts = sum(1 for c in clipped if unicodedata.category(c).startswith("P"))
    denom = max(1, len(clipped))

    math_hits = len(_MATH_RE.findall(clipped))
    code_hits = len(_CODE_RE.findall(clipped))
    reason_hits = len(_REASON_RE.findall(clipped))
    simple_hits = len(_SIMPLE_RE.findall(clipped))
    aime_hits = len(_AIME_RE.findall(clipped))
    mc_options = len(_MC_OPTION_RE.findall(clipped))
    rule_hits = len(_RULE_CHAIN_RE.findall(clipped))
    word_problem_hits = len(_WORD_PROBLEM_RE.findall(clipped))

    numeric = [
        math.log1p(n_chars),
        math.log1p(n_words),
        math.log1p(len(lines)),
        hangul / denom,
        ascii_alpha / denom,
        digits / denom,
        spaces / denom,
        puncts / denom,
        math.log1p(math_hits),
        math.log1p(code_hits),
        math.log1p(reason_hits),
        math.log1p(simple_hits),
        math.log1p(aime_hits),
        1.0 if n_chars >= 8000 else 0.0,
        1.0 if n_chars < 200 else 0.0,
        math.log1p(clipped.count("?")),
        math.log1p(clipped.count("\n\n")),
        (sum(len(w) for w in words) / max(1, n_words)),
        math.log1p(max((len(line) for line in lines), default=0)),
        math.log1p(sum(1 for w in words if w.isdigit())),
        1.0 if math_hits >= 3 else 0.0,
        1.0 if code_hits >= 2 else 0.0,
        1.0 if hangul > 0 else 0.0,
        math.log1p(clipped.count("$")),
        math.log1p(clipped.count("(")),
        math.log1p(clipped.count("[")),
        math.log1p(mc_options),
        1.0 if mc_options >= 3 and _MC_HEAD_RE.search(clipped) else 0.0,
        math.log1p(rule_hits),
        1.0 if rule_hits >= 5 else 0.0,
        1.0 if _CODE_EXEC_RE.search(clipped) else 0.0,
        1.0 if _DM_MATH_RE.search(clipped) and n_chars < 400 else 0.0,
        math.log1p(word_problem_hits),
        1.0,  # bias
    ]
    for i, v in enumerate(numeric):
        if v != 0.0:
            features[i] = float(v)

    tokens = [w.lower() for w in words[:3000]]
    counts: Dict[int, float] = {}

    def add(token: str, salt: str, weight: float = 1.0) -> None:
        index, sign = _bucket(token, salt)
        counts[index] = counts.get(index, 0.0) + sign * weight

    for t in tokens:
        add(t, "u")
    for a, b in zip(tokens, tokens[1:]):
        add(a + " " + b, "b")
    compact = re.sub(r"\s+", " ", clipped[:4000].lower())
    for i in range(0, max(0, len(compact) - 3), 2):
        add(compact[i : i + 4], "c")

    if counts:
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm > 0:
            for index, value in counts.items():
                features[NUMERIC_DIM + index] = value / norm
    return features
