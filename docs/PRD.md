# PRD: Groww HDFC Mutual Fund FAQ Chatbot

## 1. Overview

A lightweight RAG-based chatbot prototype that answers **factual questions** about 5 HDFC mutual fund pages on Groww.in. It is a hobby/learning project to test RAG capabilities — **not** a production investment advisor.

- **Website:** https://groww.in/
- **AMC:** HDFC

## 2. Goals & Non-Goals

| Goals | Non-Goals |
|---|---|
| Answer factual queries (expense ratio, lock-in, SIP min, exit load, riskometer, benchmark) | No buy/sell/hold advice |
| Show citation link in every answer | No portfolio tracking or personalized recommendations |
| Refuse opinionated questions politely | No performance claims or return comparisons |
| No PII accepted or stored | No app back-end data or third-party blog sources |

## 3. Corpus (Strict Scope)

Only these 5 Groww URLs — nothing beyond that:

| Fund Type | URL |
|---|---|
| Large-cap | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| Flexi-cap | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| ELSS | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| Small-cap | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| Hybrid | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

## 4. Architecture

```
User Question
    -> Embed (sentence-transformers/all-MiniLM-L6-v2)
    -> Query ChromaDB (top-k chunks)
    -> Build Prompt (system + retrieved context + user Q)
    -> Mistral AI API -> Answer
    -> UI (Streamlit)
```

**Key Stack:**

- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (free, lightweight)
- **Vector Store:** ChromaDB (local)
- **LLM:** Mistral AI (via API)
- **UI:** Streamlit (Groww green color theme)

### 4.1 Data Ingestion

1. Scrape the 5 URLs (publicly available page content only)
2. Chunk the content (overlap-based text splitting)
3. Embed chunks with `all-MiniLM-L6-v2`
4. Store embeddings + metadata (source URL, fund name, chunk text) in ChromaDB

### 4.2 Data Retrieval

1. Embed user question with the same model
2. Find relevant chunks from vector space (top-k similarity search)
3. Generate LLM prompt (system + retrieved context + user question)
4. Generate answer via Mistral AI API

## 5. UI Requirements

- Welcome line + 3 example questions
- Disclaimer note: *"Facts-only. No investment advice."*
- Answer <= 3 sentences with one citation link
- Footer: `"Last updated from sources: [date]"`

## 6. Edge Cases to Handle

| Edge Case | Expected Behavior |
|---|---|
| Opinion question ("Should I buy X?") | Polite refusal + educational link |
| PII in input (PAN, Aadhaar, phone) | Refuse to process, warn user |
| Performance/return question | Link to official factsheet, no computation |
| Out-of-scope question (not about these 5 funds) | "I can only answer questions about the 5 listed HDFC funds" |
| Unanswerable from corpus | "I don't have that information in my sources" |

## 7. Constraints

- **Public sources only.** No screenshots of the app back-end; no third-party blogs as sources.
- **No PII.** Do not accept/store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers.
- **No performance claims.** Don't compute/compare returns; link to the official factsheet if asked.
- **Clarity & transparency.** Keep answers <= 3 sentences; add "Last updated from sources: ".

## 8. Success Criteria

- Correctly answers factual questions with citation >= 90% of the time
- Gracefully refuses opinion/PII/performance queries 100% of the time
- Response latency < 10s (Mistral API dependent)
- Runs locally with `streamlit run app.py`

## 9. Out of Scope (Future)

- Multi-turn conversation memory
- More funds / AMCs
- Portfolio advice engine
- Authentication / user accounts