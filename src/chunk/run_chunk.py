import argparse
import json
from pathlib import Path

from src.chunk.chunker import TextChunker
from src.config import CHUNKS_DIR, RAW_DATA_DIR


def load_raw_documents(raw_dir: Path) -> list[dict]:
    documents = []
    for path in sorted(raw_dir.glob("*.json")):
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Chunking - split raw documents into retrievable chunks")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory with raw documents (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHUNKS_DIR / "chunks.json",
        help="Output chunk file (default: data/chunks/chunks.json)",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Target chunk size in characters")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Overlap between chunks in characters",
    )
    args = parser.parse_args()

    documents = load_raw_documents(args.input)
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunker.chunk_all(documents)

    per_fund: dict[str, int] = {}
    for chunk in chunks:
        per_fund[chunk.fund_category] = per_fund.get(chunk.fund_category, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for fund_category, count in per_fund.items():
        print(f"[OK] {fund_category:<12} {count:>3} chunks")
    print(f"\nSaved {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()