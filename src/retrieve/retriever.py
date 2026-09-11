from typing import Optional

import numpy as np

from src.embed.embedder import ChunkEmbedder
from src.vector_store.chroma_store import ChromaVectorStore, SearchHit

MIN_SCORE = 0.28


class Retriever:
    def __init__(
        self,
        embedder: ChunkEmbedder,
        store: ChromaVectorStore,
        top_k: int = 5,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str, where: Optional[dict] = None) -> list[SearchHit]:
        query_vector = self.embedder.embed_text(query)
        raw = self.store.query(query_vector, top_k=self.top_k * 2, where=where)
        seen_funds: set[str] = set()
        deduped: list[SearchHit] = []
        for hit in raw:
            if hit.score < self.min_score:
                continue
            if hit.fund_category in seen_funds:
                continue
            seen_funds.add(hit.fund_category)
            deduped.append(hit)
            if len(deduped) >= self.top_k:
                break
        return deduped