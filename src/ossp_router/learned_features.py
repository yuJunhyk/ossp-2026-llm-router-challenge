# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-content-only feature extraction for the learned router.

Pure standard library, deterministic (crc32 hashing). Shared verbatim by the
offline training pipeline and the container runtime. Metadata such as
``episode_id`` is never an input.

analysis/os2_features.py(예측기 재대결에 쓴 학습측 사본)와 문자 그대로
동일해야 한다 — tests/test_feature_parity.py가 동등성을 검증한다.
"""

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from typing import Dict, Tuple

HASH_DIM = 8192  # 2**13
NUMERIC_DIM = 67  # 0..33 기존 numeric(33=bias), 40..66 KFEAT 27종 (v1.7)
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



# ---------------------------------------------------------------- KFEAT (v1.7)
# K-only(K는 풀고 M은 못 푸는) 문항 판별 밀집 특징 27종 — 산술 규모·수학 구조·코드 제어흐름.
# 표준 라이브러리만 사용. 라벨 무관 결정적.
KFEAT_SLOT = 40  # numeric 인덱스 40..66 (기존 0..39 불변)
KFEAT_DIM = 27

_KF_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_KF_INT = re.compile(r"\d+")
_KF_DEC = re.compile(r"\d+\.(\d+)")
_KF_MUL = re.compile(r"\*(?!\*)|\btimes\b|\bmultiply\b|\bproduct\b", re.I)
_KF_DIV = re.compile(r"\bdivided by\b|\bdivide\b|(?<![*/])/(?![/*])", re.I)
_KF_PLACE = re.compile(
    r"\b(units|tens|hundreds|thousands|ten thousands|hundred thousands|millions|ten millions|hundred millions|billions)\s+digit",
    re.I)
_KF_PLACE_RANK = {"units": 1, "tens": 2, "hundreds": 3, "thousands": 4, "ten thousands": 5,
               "hundred thousands": 6, "millions": 7, "ten millions": 8, "hundred millions": 9, "billions": 10}
_KF_ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.I)
_KF_ORD_RANK = {w: i + 1 for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"])}
_KF_DERIV = re.compile(r"(first|second|third|fourth|fifth)\s+derivative", re.I)
_KF_BASE = re.compile(r"\bbase\s+\d+\b|\bin base\b", re.I)
_KF_PROB = re.compile(r"\bprob(?:ability)?\b.*\bsequence\b|without replacement|picked", re.I)
_KF_FRAC = re.compile(r"-?\d+/\d+")
_KF_FUNC_DEF = re.compile(r"\bdef\s+f\s*\(")
_KF_LOOP = re.compile(r"^\s*(for|while)\b", re.M)
_KF_WHILE = re.compile(r"^\s*while\b", re.M)
_KF_IF = re.compile(r"^\s*(if|elif)\b", re.M)
_KF_STRM = re.compile(r"\.(center|ljust|rjust|zfill|removesuffix|removeprefix|partition|rpartition|replace|split|rsplit|join|strip|lstrip|rstrip|find|rfind|index|count|swapcase|title|capitalize|translate|expandtabs)\(")
_KF_SLICE = re.compile(r"\[[^\[\]]*:[^\[\]]*\]")
_KF_MODOP = re.compile(r"%\s*\d|\d\s*%")
_KF_INPUT_PRED = re.compile(r"assert\s+f\(\?\?\)")
_KF_OUTPUT_PRED = re.compile(r"==\s*\?\?")
_KF_LATEX = re.compile(r"\$[^$]+\$")

KFEAT_NAMES = [
    "max_int_digits", "n_big_ints", "sum_dec_places", "max_dec_places", "mul_digit_mass",
    "n_mul", "n_div", "place_rank", "ord_rank", "deriv_order", "is_base", "is_prob_seq",
    "n_frac", "mixed_frac_dec", "n_latex", "total_digit_mass",
    "code_lines", "n_loops", "n_while", "n_if", "n_strm", "n_slice", "n_modop",
    "input_pred", "expected_len", "nest_depth", "max_arg_int_digits",
]


def _kfeat_numeric(text: str) -> list:
    t = text[:6000]
    ints = _KF_INT.findall(t)
    int_lens = [len(s.lstrip("0") or "0") for s in ints]
    decs = [len(m) for m in _KF_DEC.findall(t)]
    nums = _KF_NUM.findall(t)
    digit_mass = sum(len(n.replace("-", "").replace(".", "")) for n in nums)
    # 곱셈 피연산자 자릿수 곱 (정밀 산술 부담)
    mul_mass = 0.0
    for m in _KF_MUL.finditer(t):
        left = _KF_NUM.findall(t[max(0, m.start() - 30):m.start()])
        right = _KF_NUM.findall(t[m.end():m.end() + 30])
        if left and right:
            a = len(left[-1].replace("-", "").replace(".", ""))
            b = len(right[0].replace("-", "").replace(".", ""))
            mul_mass = max(mul_mass, a * b)
    pm = _KF_PLACE.search(t)
    om = _KF_ORD.findall(t)
    dm = _KF_DERIV.search(t)
    is_code = bool(_KF_FUNC_DEF.search(t))
    code_lines = 0; nest = 0; exp_len = 0; arg_digits = 0
    if is_code:
        body = t.split("assert")[0]
        lines = [l for l in body.splitlines() if l.strip()]
        code_lines = len(lines)
        nest = max((len(l) - len(l.lstrip(" "))) // 4 for l in lines) if lines else 0
        tail = t.split("assert", 1)[1] if "assert" in t else ""
        if "==" in tail:
            lhs, rhs = tail.split("==", 1)
            exp_len = len(rhs.strip()) if "??" not in rhs else len(lhs.strip())
            args = lhs if "??" not in lhs else rhs
            arg_digits = max((len(s) for s in _KF_INT.findall(args)), default=0)
    frac = len(_KF_FRAC.findall(t))
    return [
        math.log1p(max(int_lens, default=0)),
        math.log1p(sum(1 for l in int_lens if l >= 5)),
        math.log1p(sum(decs)),
        math.log1p(max(decs, default=0)),
        math.log1p(mul_mass),
        math.log1p(len(_KF_MUL.findall(t))),
        math.log1p(len(_KF_DIV.findall(t))),
        float(_KF_PLACE_RANK.get(pm.group(1).lower(), 0)) if pm else 0.0,
        float(max((_KF_ORD_RANK[w.lower()] for w in om), default=0)),
        float(_KF_ORD_RANK.get(dm.group(1).lower(), 0)) if dm else 0.0,
        1.0 if _KF_BASE.search(t) else 0.0,
        1.0 if _KF_PROB.search(t) else 0.0,
        math.log1p(frac),
        1.0 if frac and decs else 0.0,
        math.log1p(len(_KF_LATEX.findall(t))),
        math.log1p(digit_mass),
        math.log1p(code_lines),
        math.log1p(len(_KF_LOOP.findall(t))) if is_code else 0.0,
        math.log1p(len(_KF_WHILE.findall(t))) if is_code else 0.0,
        math.log1p(len(_KF_IF.findall(t))) if is_code else 0.0,
        math.log1p(len(_KF_STRM.findall(t))) if is_code else 0.0,
        math.log1p(len(_KF_SLICE.findall(t))) if is_code else 0.0,
        math.log1p(len(_KF_MODOP.findall(t))) if is_code else 0.0,
        1.0 if is_code and _KF_INPUT_PRED.search(t) else 0.0,
        math.log1p(exp_len),
        float(nest),
        math.log1p(arg_digits),
    ]

# K 비용 폭주 가드 — 7자리 이상 정수와 함께 다음이 나타나면 K(think) 승급 후보에서 제외한다.
# (a) 소수/합성수/소인수분해 어휘: K 비용이 중앙값의 ~33배로 폭주하고 비용 예측기가 12~14배 과소 예측.
# (b) 2차 이상 다항식 방정식: 같은 줄에서 거듭제곱(`x**2`·`x^3`) 뒤에 `= 0`이 오는 문장
#     ("… = 0.", "… = 0. What is x?", "… = 0 for j." — `= 0.5` 같은 소수는 제외). 공개 train에서
#     (a)에 걸리지 않는 해당 문항은 6건이고 L/M/K 실측 점수 평균 0/0/0.08, K 비용 합계 4.1 credits —
#     승급 이득이 없는 폭탄 유형. train 전용 교차검증 게이트(analysis/cv_gate.py)에서 점수 손실 없이
#     소표본 예산 통과율을 올렸다.
_KGUARD_PAT = re.compile(r"\b(prime|composite|prime factors?|factors? of)\b", re.I)
_KGUARD_BIG = re.compile(r"\d{7,}")
_KGUARD_POLY_EQ0 = re.compile(r"(?:\*\*|\^)\s*[2-9].*?=\s*0(?![.,]?\d)")


def k_guard(text: str) -> bool:
    """True면 axk1-think 승급 금지 (예측 K 점수를 M 점수로 대체)."""
    t = text[:6000]
    if not _KGUARD_BIG.search(t):
        return False
    if _KGUARD_PAT.search(t):
        return True
    return bool(_KGUARD_POLY_EQ0.search(t))


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
    for i, v in enumerate(_kfeat_numeric(text)):
        if v != 0.0:
            features[KFEAT_SLOT + i] = float(v)

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
