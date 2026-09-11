import json
from pathlib import Path

import pytest

from src.config import RAW_DATA_DIR

EXPECTED_FACT_LABELS = [
    "Expense ratio (Direct)",
    "Base expense ratio",
    "Exit load",
    "Minimum SIP investment",
    "Benchmark",
    "Riskometer",
]


def load_documents() -> list[dict]:
    documents = []
    for path in sorted(RAW_DATA_DIR.glob("*.json")):
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    return documents


def test_five_documents_saved() -> None:
    assert len(load_documents()) == 5


def test_corpus_urls_are_from_prd_scope() -> None:
    allowed_host = "https://groww.in/mutual-funds/"
    for document in load_documents():
        assert document["url"].startswith(allowed_host)


def test_every_document_has_required_fields() -> None:
    required = ["url", "fund_name", "fund_category", "crawled_at", "title", "facts", "content", "text"]
    for document in load_documents():
        for field in required:
            assert document.get(field), f"{document['fund_category']} missing {field}"


def test_facts_cover_prd_questions() -> None:
    documents = {d["fund_category"]: d for d in load_documents()}
    assert documents.keys() >= {"Large Cap", "Flexi Cap", "ELSS", "Small Cap", "Hybrid"}
    for category, document in documents.items():
        facts = document["facts"]
        for label in EXPECTED_FACT_LABELS:
            assert label in facts, f"{category} facts missing '{label}'"
        assert "No structured facts available" not in facts


def test_elss_specific_facts() -> None:
    documents = {d["fund_category"]: d for d in load_documents()}
    facts = documents["ELSS"]["facts"]
    assert "Lock-in period: 3 year(s)" in facts
    assert "Minimum SIP investment: 500" in facts
    assert "Exit load: Nil" in facts


def test_large_cap_specific_facts() -> None:
    documents = {d["fund_category"]: d for d in load_documents()}
    facts = documents["Large Cap"]["facts"]
    assert "Benchmark: NIFTY 100 Total Return Index" in facts
    assert "Benchmark code: NIFTY 100 TRI" in facts
    assert "Minimum SIP investment: 100" in facts


def test_text_starts_with_facts_block() -> None:
    for document in load_documents():
        assert document["text"].startswith(document["facts"])


def test_content_is_meaningful() -> None:
    for document in load_documents():
        assert len(document["content"]) > 500, f"{document['fund_category']} content too short"