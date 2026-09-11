from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from src.chunk.chunker import Chunk
from src.config import VECTOR_DB_DIR

COLLECTION_NAME = "hdfc_funds"
DEFAULT_TOP_K = 5
INDEX_BATCH_SIZE = 50


class SearchHit(BaseModel):
    chunk_id: str
    url: str
    fund_name: str
    fund_category: str
    chunk_text: str
    score: float


class ChromaVectorStore:
    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = COLLECTION_NAME) -> None:
        self.persist_dir = Path(persist_dir or VECTOR_DB_DIR)
        self.collection_name = collection_name
        self._client = self._new_client()
        self.collection = self._get_or_create()

    def _new_client(self):
        import chromadb

        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _get_or_create(self):
        return self._client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create()

    def index(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.chunk_text for chunk in chunks]
        metadatas = [
            {
                "url": chunk.url,
                "fund_name": chunk.fund_name,
                "fund_category": chunk.fund_category,
            }
            for chunk in chunks
        ]
        for start in range(0, len(chunks), INDEX_BATCH_SIZE):
            end = start + INDEX_BATCH_SIZE
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def query(self, embedding: np.ndarray, top_k: int = DEFAULT_TOP_K, where: Optional[dict] = None) -> list[SearchHit]:
        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=where,
        )
        hits = []
        for i, chunk_id in enumerate(result["ids"][0]):
            metadata = result["metadatas"][0][i]
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    url=metadata["url"],
                    fund_name=metadata["fund_name"],
                    fund_category=metadata["fund_category"],
                    chunk_text=result["documents"][0][i],
                    score=min(1.0, max(0.0, 1.0 - result["distances"][0][i])),
                )
            )
        return hits

    def count(self) -> int:
        return self.collection.count()