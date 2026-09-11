import numpy as np

from src.embed.embedder import ChunkEmbedder
from src.vector_store.chroma_store import ChromaVectorStore

embedder = ChunkEmbedder(script_dir=".")
store = ChromaVectorStore()

queries = [
    ("elss expense", "What is the expense ratio of HDFC ELSS fund?"),
    ("large benchmark", "What is the benchmark for HDFC Large Cap fund?"),
    ("elss lock-in", "Does HDFC ELSS have a lock-in period?"),
    ("small min sip", "What is the minimum SIP amount for HDFC Small Cap fund?"),
    ("hybrid exit load", "What is the exit load for HDFC Balanced Advantage fund?"),
]
for label, q in queries:
    v = embedder.embed_text(q)
    hits = store.query(v, top_k=1)
    h = hits[0]
    print(f"[{label}] top1 {h.score:.3f} {h.fund_name} | {h.chunk_id}")
