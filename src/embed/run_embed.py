import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.chunk.chunker import Chunk
from src.config import CHUNKS_DIR, EMBEDDINGS_DIR
from src.embed.embedder import EMBEDDING_DIM, MODEL_NAME, ChunkEmbedder


def load_chunks(chunks_file: Path) -> list[Chunk]:
    data = json.loads(chunks_file.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Embedding - embed chunks with all-MiniLM-L6-v2")
    parser.add_argument(
        "--input",
        type=Path,
        default=CHUNKS_DIR / "chunks.json",
        help="Chunk file (default: data/chunks/chunks.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EMBEDDINGS_DIR,
        help="Embedding output dir (default: data/embeddings)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    chunks = load_chunks(args.input)
    embedder = ChunkEmbedder(batch_size=args.batch_size)
    texts = [chunk.chunk_text for chunk in chunks]
    vectors = embedder.embed_texts(texts)

    args.output.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output / "embeddings.npy"
    ids_path = args.output / "chunk_ids.json"
    meta_path = args.output / "meta.json"

    np.save(embeddings_path, vectors)
    ids_path.write_text(
        json.dumps([chunk.chunk_id for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "model": embedder.model_name,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_count": len(chunks),
        "batch_size": args.batch_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Embedded {vectors.shape[0]} chunks -> {embeddings_path} (dtype {vectors.dtype}, dim {vectors.shape[1]})")
    print(f"Chunk ids  -> {ids_path}")
    print(f"Meta       -> {meta_path}")


if __name__ == "__main__":
    main()