from typing import Optional

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 32


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class ChunkEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[object] = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]