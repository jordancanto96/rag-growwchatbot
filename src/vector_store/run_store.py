import argparse
import json
from pathlib import Path

import numpy as np

from src.chunk.chunker import Chunk
from src.config import CHUNKS_DIR, EMBEDDINGS_DIR, VECTOR_DB_DIR
from src.vector_store.chroma_store import COLLECTION_NAME, ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Vector Store - index chunks into local ChromaDB")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=CHUNKS_DIR / "chunks.json",
        help="Chunk file (default: data/chunks/chunks.json)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=EMBEDDINGS_DIR / "embeddings.npy",
        help="Embeddings matrix (default: data/embeddings/embeddings.npy)",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=EMBEDDINGS_DIR / "chunk_ids.json",
        help="Chunk id order file (default: data/embeddings/chunk_ids.json)",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Chroma collection name",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    args = parser.parse_args()

    chunks = [Chunk(**item) for item in json.loads(args.chunks.read_text(encoding="utf-8"))]
    expected_ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
    actual_ids = [chunk.chunk_id for chunk in chunks]
    if actual_ids != expected_ids:
        raise SystemExit(f"chunk order mismatch: chunks.json vs {args.ids_file}")
    vectors = np.load(args.embeddings)
    if vectors.shape != (len(chunks), vectors.shape[1]):
        raise SystemExit(f"embedding shape {vectors.shape} does not match {len(chunks)} chunks")

    store = ChromaVectorStore(collection_name=args.collection)
    if args.rebuild:
        store.reset()

    existing = store.count()
    if existing:
        print(f"Collection '{args.collection}' exists with {existing} vectors; re-indexing current corpus")
    store.index(chunks, vectors)

    indexed = store.collection.get()
    by_fund: dict[str, int] = {}
    for chunk_id, meta in zip(indexed["ids"], indexed["metadatas"]):
        by_fund[meta["fund_category"]] = by_fund.get(meta["fund_category"], 0) + 1
    for fund, count in sorted(by_fund.items()):
        print(f"[OK] {fund:<12} {count:>3} vectors indexed")

    print(f"Collection '{args.collection}' now has {store.count()} vectors in {store.persist_dir}")


if __name__ == "__main__":
    main()