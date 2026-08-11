"""T1: EDA + tier별 oracle 상한 산출.

공식 비용 정책(configs/routing-policy.v1.json)으로 문항별 비용을 계산하고,
λ-스윕(라그랑주 완화 + 이분 탐색)으로 예산 제약 하 최적 배정의 상한을 구한다.
"""

import json
import statistics
from decimal import Decimal

REPO = "/Users/yujunhyeog/ossp-router"
MODELS = ["ax31-light", "ax31", "axk1-think"]
RATES = {  # (input_rate, output_rate) per 1M tokens, fixed=0
    "ax31-light": (Decimal("1"), Decimal("4")),
    "ax31": (Decimal("2.127"), Decimal("8.509")),
    "axk1-think": (Decimal("6.565"), Decimal("26.260")),
}
UNIT = Decimal(1_000_000)
TIERS = {
    "fast": (Decimal("1.25"), Decimal("0.4")),
    "balanced": (Decimal("2.0"), Decimal("0.3")),
    "premium": (Decimal("4.0"), Decimal("0.3")),
}


def load_split(split):
    outcomes = json.load(open(f"{REPO}/data/{split}/outcomes.json"))
    episodes = []
    for ep in outcomes["episodes"]:
        row = {}
        for m in MODELS:
            o = ep["models"][m]
            score = float(o["score"])
            cost = (
                Decimal(o["input_tokens"]) * RATES[m][0]
                + Decimal(o["output_tokens"]) * RATES[m][1]
            ) / UNIT
            row[m] = (score, cost, o["input_tokens"], o["output_tokens"])
        episodes.append(row)
    return episodes


def oracle(episodes, multiplier):
    """λ 이분 탐색으로 예산 제약 하 최적(LP 완화) 배정."""
    light_cost = sum(e["ax31-light"][1] for e in episodes)
    limit = light_cost * multiplier

    def assign(lam):
        total_s, total_c, counts = 0.0, Decimal(0), {m: 0 for m in MODELS}
        for e in episodes:
            best = max(MODELS, key=lambda m: (e[m][0] - lam * float(e[m][1]), -e[m][1]))
            total_s += e[best][0]
            total_c += e[best][1]
            counts[best] += 1
        return total_s, total_c, counts

    s0, c0, n0 = assign(0.0)
    if c0 <= limit:
        return s0 / len(episodes), c0 / light_cost, n0
    lo, hi = 0.0, 1e9
    best_feasible = None
    for _ in range(80):
        mid = (lo + hi) / 2
        s, c, n = assign(mid)
        if c <= limit:
            best_feasible = (s, c, n)
            hi = mid
        else:
            lo = mid
    s, c, n = best_feasible
    return s / len(episodes), c / light_cost, n


for split in ("train", "dev"):
    eps = load_split(split)
    N = len(eps)
    print(f"\n{'='*62}\n[{split}] {N} 문항\n{'='*62}")

    # --- score 구조 ---
    means = {m: statistics.mean(e[m][0] for e in eps) for m in MODELS}
    print("모델별 평균 score:", {m: round(v, 4) for m, v in means.items()})
    binary = sum(all(e[m][0] in (0.0, 1.0) for m in MODELS) for e in eps)
    print(f"score가 0/1 이진인 문항: {binary}/{N}")
    l, m_, k = "ax31-light", "ax31", "axk1-think"
    up_m = [e[m_][0] - e[l][0] for e in eps]
    up_k = [e[k][0] - e[l][0] for e in eps]
    print(
        f"uplift(L→M): 평균 {statistics.mean(up_m):+.4f} | 양수 {sum(u>0 for u in up_m)} / 0 {sum(u==0 for u in up_m)} / 음수 {sum(u<0 for u in up_m)}"
    )
    print(
        f"uplift(L→K): 평균 {statistics.mean(up_k):+.4f} | 양수 {sum(u>0 for u in up_k)} / 0 {sum(u==0 for u in up_k)} / 음수 {sum(u<0 for u in up_k)}"
    )
    print(f"K<M인 문항(역전): {sum(e[k][0] < e[m_][0] for e in eps)}")

    # --- 비용 구조 ---
    for m in MODELS:
        costs = sorted(float(e[m][1]) for e in eps)
        outs = sorted(e[m][3] for e in eps)
        print(
            f"{m:<12} 비용 평균 {statistics.mean(costs):.6f}  p50 {costs[N//2]:.6f}  p90 {costs[int(N*0.9)]:.6f}  p99 {costs[int(N*0.99)]:.6f} | 출력토큰 p50 {outs[N//2]}  p99 {outs[int(N*0.99)]}"
        )
    lc = sum(e[l][1] for e in eps)
    print(
        f"all-light 총비용: {float(lc):.4f} | all-M 비율 {float(sum(e[m_][1] for e in eps)/lc):.3f} | all-K 비율 {float(sum(e[k][1] for e in eps)/lc):.3f}"
    )

    # --- oracle ---
    print(f"\n{'tier':<10}{'oracle':>9}{'비용비율':>9}   배정(L/M/K)")
    weighted = 0.0
    anchors = {"train": None, "dev": 0.619318}
    for tier, (mult, w) in TIERS.items():
        s, ratio, n = oracle(eps, mult)
        weighted += float(w) * s
        print(f"{tier:<10}{s:>9.6f}{float(ratio):>9.4f}   {n[l]}/{n[m_]}/{n[k]}")
    print(f"가중 oracle 최종점수: {weighted:.6f}")
    if split == "dev":
        al, hr = 0.619318, 0.695369
        print(f"\nheadroom: oracle {weighted:.4f} − all-light {al} = {weighted-al:.4f}")
        print(
            f"hash-regex 회수율: ({hr}−{al}) / ({weighted:.4f}−{al}) = {(hr-al)/(weighted-al)*100:.1f}%"
        )
