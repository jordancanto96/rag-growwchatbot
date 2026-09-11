import json

import numpy as np

from src.chunk.chunker import Chunk
from src.config import CHUNKS_DIR, EMBEDDINGS_DIR
from src.vector_store.chroma_store import ChromaVectorStore

COLLECTION_NAME = "hdfc_funds"
EXPECTED_COUNT = 81


def build_chunks() -> list[Chunk]:
    data = json.loads((CHUNKS_DIR / "chunks.json").read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


def load_vectors() -> np.ndarray:
    return np.load(EMBEDDINGS_DIR / "embeddings.npy")


def make_store(reuse: bool = False) -> ChromaVectorStore:
    store = ChromaVectorStore(collection_name=COLLECTION_NAME)
    if reuse:
        return store
    store.reset()
    store.index(build_chunks(), load_vectors())
    return store


def test_collection_has_expected_count() -> None:
    store = make_store()
    assert store.count() == EXPECTED_COUNT


def test_query_returns_hits_with_metadata_and_score() -> None:
    store = make_store()
    vectors = load_vectors()
    hits = store.query(vectors[0], top_k=5)
    assert len(hits) == 5
    for hit in hits:
        assert hit.chunk_id.startswith("hdfc-")
        assert hit.url.startswith("https://groww.in/mutual-funds/")
        assert hit.fund_name
        assert hit.fund_category
        assert hit.chunk_text
        assert 0.0 <= hit.score <= 1.0


def test_query_top_hit_is_self() -> None:
    store = make_store()
    vectors = load_vectors()
    chunk_ids = json.loads((EMBEDDINGS_DIR / "chunk_ids.json").read_text(encoding="utf-8"))
    hits = store.query(vectors[0], top_k=3)
    assert hits[0].chunk_id == chunk_ids[0]


def test_metadata_filter_restricts_to_fund() -> None:
    store = make_store()
    vectors = load_vectors()
    hits = store.query(vectors[0], top_k=50, where={"fund_category": "ELSS"})
    assert hits
    for hit in hits:
        assert hit.fund_category == "ELSS"


def test_scores_are_ordered_descending() -> None:
    store = make_store()
    vectors = load_vectors()
    hits = store.query(vectors[0], top_k=10)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_number_of_upserted_vectors_matches_chunks() -> None:
    chunks = build_chunks()
    store = make_store()
    stored = store.collection.get()
    assert len(stored["ids"]) == len(chunks)
    assert set(stored["ids"]) == {chunk.chunk_id for chunk in chunks}


def test_embedding_ids_align_with_chunks_file() -> None:
    chunk_ids = [chunk.chunk_id for chunk in build_chunks()]
    saved_ids = json.loads((EMBEDDINGS_DIR / "chunk_ids.json").read_text(encoding="utf-8"))
    assert chunk_ids == saved_ids
    assert len(load_vectors()) == len(chunk_ids)


def test_rebuild_collection_idempotent() -> None:
    store = make_store()
    store.index(build_chunks(), load_vectors())
    assert store.count() == EXPECTED_COUNT