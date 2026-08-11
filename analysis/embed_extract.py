# SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
# SPDX-License-Identifier: Apache-2.0

"""train/dev 프롬프트 임베딩 추출 — 오프라인 실험용 캐시 생성 (학습 전용).

후보 모델 (둘 다 open-weight, 상업적 이용·재배포 허용 라이선스):
- intfloat/multilingual-e5-small (MIT) — "query: " 프리픽스 규약
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input

MODELS = {
    "e5small": ("intfloat/multilingual-e5-small", "query: "),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "build/embeddings")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ("train", "dev"):
        inputs = load_input(REPO / f"data/materialized/{split}/inputs.json")
        splits[split] = [episode_text(ep) for ep in inputs.episodes]
        print(f"{split}: {len(splits[split])}문항")

    for key, (model_name, prefix) in MODELS.items():
        print(f"\n[{key}] {model_name} 로드 중 ...")
        model = SentenceTransformer(model_name)
        for split, texts in splits.items():
            batch = [prefix + t for t in texts]
            emb = model.encode(
                batch,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out = args.out_dir / f"{key}-{split}.npy"
            np.save(out, emb.astype(np.float32))
            print(f"  {split}: {emb.shape} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
