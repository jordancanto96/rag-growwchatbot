# Architecture: Groww HDFC Mutual Fund FAQ Chatbot (RAG Prototype)

## 1. Architectural Overview

A single-machine RAG pipeline: **Ingest** (load -> chunk -> embed -> store) and **Serve** (embed query -> retrieve -> prompt -> answer), fronted by a Streamlit UI.

```
┌──────────────┐      ┌───────────────────────────────────────────────────┐
│  Scraper     │─────▶│                 INGESTION PIPELINE                │
│ (5 Groww URLs)│      │  Load ──▶ Chunk ──▶ Embed ──▶ Vector Store        │
└──────────────┘      └───────────────────────────┬───────────────────────┘
                                                   │ persist
                                                   ▼
┌────────────────┐      ┌───────────────────────────────────────────────────┐
│  Streamlit UI  │─────▶│                  SERVE PIPELINE                  │
│  user question  │      │  Embedded query ──▶ Retrieve ──▶ Prompt ──▶ Answer│
└────────────────┘      └───────────────────────────────────────────────────┘
        ▲                            │
        └──── answer + citation ─────┘
```

**Components:** Streamlit UI | Sentence-Transformers (`all-MiniLM-L6-v2`) | ChromaDB | Mistral AI API.

---

## 2. Phase 1 — Data Loading

**Input:** Exactly the 5 Groww URLs from the PRD corpus (strict scope, nothing else).

| URL | Fund |
|---|---|
| `.../hdfc-large-cap-fund-direct-growth` | Large Cap |
| `.../hdfc-equity-fund-direct-growth` | Flexi-cap (legacy slug) |
| `.../hdfc-elss-tax-saver-fund-direct-plan-growth` | ELSS |
| `.../hdfc-small-cap-fund-direct-growth` | Small Cap |
| `.../hdfc-balanced-advantage-fund-direct-growth` | Hybrid |

**Steps:**
1. Fetch the 5 pages (HTTP GET with desktop user-agent).
2. Extract main article content only (title, key facts, body) — drop nav/ads/scripts/footers.
3. Normalise text (strip whitespace, HTML entities, remove empty sections).
4. Attach metadata per document: `{ url, fund_name, fund_category, crawled_at }`.

**Rules (from PRD):** public sources only; no back-end screenshots; no third-party blogs.

**Output:** 5 cleaned documents with metadata.

---

## 3. Phase 2 — Chunking

**Goal:** Break each document into retrievable units small enough for a good LLM context and precise citations.

**Strategy:**
1. Text splitter with overlap (e.g. recursive character splitter, ~500–800 char chunks, ~10–20% overlap).
2. Semantic boundaries preferred: split on section headings / bullet lists / tables first; fall back to character level.
3. Each chunk carries inherited metadata: `{ chunk_id, url, fund_name, chunk_index, chunk_text }`.

**Output:** N chunks across the 5 funds (all originating from PRD corpus only).

---

## 4. Phase 3 — Embedding

**Model (PRD-specified):** `sentence-transformers/all-MiniLM-L6-v2` — free, lightweight, 384-dim.

**Steps:**
1. Load model once at ingestion/serve start.
2. Batch-embed all chunks (batches of ~32 to keep memory low).
3. Normalise embeddings (L2) for cosine similarity.
4. Cache the model in memory to avoid reload per query.

**Output:** A vector per chunk (384-dim, float32).

---

## 5. Phase 4 — Vector Store (ChromaDB)

**Persistent local ChromaDB collection** `hdfc_funds`:

| Chroma element | Content |
|---|---|
| `ids` | `chunk_id` (e.g. `hdfc-large-cap-0042`) |
| `embeddings` | 384-dim vectors from Phase 3 |
| `documents` | chunk text |
| `metadatas` | url, fund_name, fund_category |

**Steps:**
1. Create/reset collection `hdfc_funds` on rebuild (`--rebuild` flag).
2. Insert chunk batches.
3. Store collection on disk (`./chroma_db`) for reuse across restarts.

**Constraint:** corpus size is fixed (5 pages) — collection is small, no scaling concerns.

**Output:** Searchable local vector store.

---

## 6. Phase 5 — Retrieval Logic

**Query path:**
1. Embed the user question with the same `all-MiniLM-L6-v2` model.
2. Cosine similarity search in ChromaDB → return **top-k** (e.g. k=5) chunks with scores.
3. Optionally: metadata filter by fund name if the question names one (e.g. "ELSS lock-in" -> ELSS only).
4. Deduplicate by fund_name to avoid one fund flooding the context.

**Prompt assembly:**
```
System: You are a facts-only assistant on the 5 HDFC Groww fund pages.
        Answer in <=3 sentences. Cite exactly ONE source URL.
        Never advise buy/sell. Never compute returns.
        If no chunk answers: say "I don't have that information in my sources."

Context: [top-k chunks, each labeled with its url]

User: [question]
```

**Answer generation:** Mistral AI API (from PRD).

**Guards (from PRD edge cases):**
- Opinion questions -> polite refusal + educational link.
- PII detected -> refuse, warn (check against PAN/Aadhaar/phone/email patterns).
- Performance/return ask -> link official factsheet, no computation.
- Out-of-scope topic -> "I can only answer questions about the 5 listed HDFC funds."

**Output:** Answer (<=3 sentences) + citation URL + "Last updated from sources: [date]".

---

## 7. Phase 6 — Retrieval Testing

**Purpose:** verify retrieval quality before trusting answers.

**Test suites:**

1. **Golden-set QA (recall @k)**
   - Build a hand-written set of question -> expected source URL pairs (from PRD example questions: expense ratio, ELSS lock-in, min SIP, exit load, riskometer/benchmark, capital-gains statement).
   - For each: retrieve top-k and assert the correct fund's pages appears.
   - Metric: **Recall@k >= 0.9**.

2. **Relevance checks (precision)**
   - Assert the top-k chunks are same-fund as the question (no cross-fund drift).
   - Metric: **Precision@k >= 0.9**.

3. **Edge-case / guard tests (from PRD)**
   - Opinion: "Should I buy HDFC Large Cap?" -> refusal, no returns.
   - PII: "my PAN is ABCD12345E" -> refused, no storage.
   - Performance: "What was the return last year?" -> factsheet link, no number.
   - Out-of-scope: "Tell me about SBI fund" -> out-of-scope message.
   - Unanswerable: question not in any chunk -> "not in sources".

4. **Answer sanity (end-to-end)**
   - Every answer <=3 sentences, has exactly 1 citation URL, ends with "Last updated from sources: ".
   - Assert refusal responses never contain numerical returns.

**Output:** test report with pass/fail per suite; gate any UI release on retrieval tests passing.

---

## 8. Summary of Flow

| Phase | Component | Input | Output |
|---|---|---|---|
| 1. Load | Scraper | 5 PRD URLs | 5 cleaned docs + metadata |
| 2. Chunk | Splitter | docs | N chunks |
| 3. Embed | MiniLM-L6-v2 | chunks | 384-d vectors |
| 4. Store | ChromaDB | vectors + docs | persistent collection |
| 5. Retrieve | ChromaDB + Mistral | question | answer + citation |
| 6. Test | Test harness | golden set + edge cases | pass/fail report |

---

## 9. Tech Stack (from PRD)

| Layer | Choice |
|---|---|
| Scraping | `requests` + `BeautifulSoup` (or `trafilatura`) |
| Chunking | recursive text splitter (e.g. `langchain-text-splitters` or custom) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (local persistent) |
| LLM | Mistral AI API |
| UI | Streamlit (Groww green theme) |
| Config | `.env` for `MISTRAL_API_KEY` and corpus URLs |