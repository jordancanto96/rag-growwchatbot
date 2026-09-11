import json

import numpy as np

from src.config import CHUNKS_DIR, EMBEDDINGS_DIR

EMBEDDING_DIM = 384


def load_embeddings() -> tuple[np.ndarray, list[str], dict]:
    vectors = np.load(EMBEDDINGS_DIR / "embeddings.npy")
    ids = json.loads((EMBEDDINGS_DIR / "chunk_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((EMBEDDINGS_DIR / "meta.json").read_text(encoding="utf-8"))
    return vectors, ids, meta


def test_embedding_artifacts_exist() -> None:
    assert (EMBEDDINGS_DIR / "embeddings.npy").exists()
    assert (EMBEDDINGS_DIR / "chunk_ids.json").exists()
    assert (EMBEDDINGS_DIR / "meta.json").exists()


def test_shape_and_dtype() -> None:
    vectors, ids, _ = load_embeddings()
    assert vectors.shape == (len(ids), EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_vectors_are_l2_normalised() -> None:
    vectors, _, _ = load_embeddings()
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_aligns_with_chunk_ids() -> None:
    vectors, ids, _ = load_embeddings()
    chunks = json.loads((CHUNKS_DIR / "chunks.json").read_text(encoding="utf-8"))
    saved_ids = [chunk["chunk_id"] for chunk in chunks]
    assert ids == saved_ids
    assert len(ids) == len(chunks)


def test_meta_reports_model() -> None:
    _, _, meta = load_embeddings()
    assert meta["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta["embedding_dim"] == EMBEDDING_DIM
    assert meta["chunk_count"] == len(json.loads((CHUNKS_DIR / "chunks.json").read_text(encoding="utf-8")))


def test_l2_normalise_handles_zero_vector() -> None:
    from src.embed.embedder import l2_normalise

    matrix = np.zeros((2, 4), dtype=np.float32)
    result = l2_normalise(matrix)
    assert np.allclose(result, matrix)
    assert np.all(np.linalg.norm(result, axis=1) == 0.0)


def test_l2_normalise_scales_rows() -> None:
    from src.embed.embedder import l2_normalise

    matrix = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    result = l2_normalise(matrix)
    assert np.allclose(np.linalg.norm(result, axis=1), [1.0, 1.0], atol=1e-5)
    assert result[0, 0] == 0.6 and result[0, 1] == 0.8


def test_distinct_funds_sections_are_not_identical() -> None:
    vectors, ids, _ = load_embeddings()
    by_id = dict(zip(ids, vectors))
    elss_facts = np.asarray(by_id["hdfc-elss-0000"])
    hybrid_exit_load = np.asarray(by_id["hdfc-hybrid-0043"])
    assert float(np.dot(elss_facts, hybrid_exit_load)) < 0.9