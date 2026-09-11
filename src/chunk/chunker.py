from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEPARATORS = ["\n\n", "\n", "|", "·", "•", ". ", " "]


class Chunk(BaseModel):
    chunk_id: str
    url: str
    fund_name: str
    fund_category: str
    chunk_index: int
    chunk_text: str


class TextChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[list[str]] = None,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
            length_function=len,
            keep_separator=True,
        )

    def chunk_document(self, document: dict) -> list[Chunk]:
        text = document.get("text") or ""
        url = document["url"]
        fund_name = document["fund_name"]
        fund_category = document["fund_category"]
        pieces = self.splitter.split_text(text)
        slug = fund_category.lower().replace(" ", "-")
        return [
            Chunk(
                chunk_id=f"hdfc-{slug}-{index:04d}",
                url=url,
                fund_name=fund_name,
                fund_category=fund_category,
                chunk_index=index,
                chunk_text=piece,
            )
            for index, piece in enumerate(pieces)
        ]

    def chunk_all(self, documents: list[dict]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks