import json

from src.chunk.chunker import TextChunker
from src.config import CHUNKS_DIR, RAW_DATA_DIR


def load_chunks() -> list[dict]:
    path = CHUNKS_DIR / "chunks.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_documents() -> list[dict]:
    documents = []
    for path in sorted(RAW_DATA_DIR.glob("*.json")):
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    return documents


def test_chunk_ids_look_sane() -> None:
    for chunk in load_chunks():
        slug = chunk["fund_category"].lower().replace(" ", "-")
        expected = f"hdfc-{slug}-{chunk['chunk_index']:04d}"
        assert chunk["chunk_id"] == expected


def test_every_fund_has_chunks() -> None:
    categories = {c["fund_category"] for c in load_chunks()}
    assert categories >= {"Large Cap", "Flexi Cap", "ELSS", "Small Cap", "Hybrid"}


def test_chunks_carry_required_metadata() -> None:
    required = ["chunk_id", "url", "fund_name", "fund_category", "chunk_index", "chunk_text"]
    for chunk in load_chunks():
        for field in required:
            assert field in chunk, f"{chunk['chunk_id']} missing {field}"
            if field != "chunk_index":
                assert chunk[field], f"{chunk['chunk_id']} empty {field}"
        assert chunk["url"].startswith("https://groww.in/mutual-funds/")


def test_chunk_lengths_within_limits() -> None:
    for chunk in load_chunks():
        assert 1 <= len(chunk["chunk_text"]) <= 800, chunk["chunk_id"]


def test_facts_block_preserved_as_chunk_zero() -> None:
    documents = {d["fund_category"]: d for d in load_raw_documents()}
    by_fund: dict[str, list[dict]] = {}
    for chunk in load_chunks():
        by_fund.setdefault(chunk["fund_category"], []).append(chunk)
    for category, document in documents.items():
        first = next(c for c in by_fund[category] if c["chunk_index"] == 0)
        assert first["chunk_text"].startswith("Fund: "), category
        for line in document["facts"].splitlines():
            assert line in first["chunk_text"], f"{category} facts line lost: {line}"


def test_prd_answerable_facts_survive_chunking() -> None:
    chunks = load_chunks()
    elss = " ".join(c["chunk_text"] for c in chunks if c["fund_category"] == "ELSS")
    assert "Lock-in period: 3 year(s)" in elss
    assert "Exit load: Nil" in elss
    assert "Minimum SIP investment: 500" in elss
    large = " ".join(c["chunk_text"] for c in chunks if c["fund_category"] == "Large Cap")
    assert "Benchmark: NIFTY 100 Total Return Index" in large
    assert "Riskometer: Very High risk" in large


def test_chunking_is_deterministic() -> None:
    documents = load_raw_documents()
    chunker = TextChunker()
    first = [c.model_dump() for c in chunker.chunk_all(documents)]
    second = [c.model_dump() for c in chunker.chunk_all(documents)]
    assert first == second


def test_rechunk_matches_saved_artifacts() -> None:
    documents = load_raw_documents()
    chunker = TextChunker()
    fresh = [c.model_dump() for c in chunker.chunk_all(documents)]
    assert fresh == load_chunks()