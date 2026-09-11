"""Phase 7 — Streamlit frontend for the HDFC fund RAG assistant (Groww green).

Run:
    streamlit run app.py

Wires the offline-first serve pipeline:
    question -> QueryGuard -> Retriever -> RAGAssistant.answer -> RAGAnswer

Works out of the box with the offline fallback generator; optionally upgrades to
Mistral AI once MISTRAL_API_KEY is present in the root .env.
"""

import os

import streamlit as st

from src.config import CORPUS, VECTOR_DB_DIR
from src.embed.embedder import ChunkEmbedder
from src.retrieve.rag import RAGAssistant, MistralAnswerGenerator, UnavailableAnswerGenerator
from src.retrieve.retriever import Retriever
from src.vector_store.chroma_store import ChromaVectorStore

GROWW_GREEN = "#00B386"
GROWW_GREEN_DARK = "#00967A"

st.set_page_config(
    page_title="HDFC Funds Assistant",
    page_icon="🟢",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0f1115; }}
    .block-container {{ padding-top: 2.2rem; }}
    h1, h2, h3 {{ color: {GROWW_GREEN} !important; font-weight: 700; }}
    .groww-header {{ color: #e7f6f0; margin-bottom: 0.2rem; }}
    .groww-sub {{ color: #9aa7b4; font-size: 0.92rem; margin-bottom: 1.4rem; }}
    .citation {{ background: #101a17; border-left: 3px solid {GROWW_GREEN};
                  padding: 0.65rem 0.9rem; border-radius: 6px; margin-top: 0.6rem;
                  font-size: 0.85rem; }}
    .refusal {{ background: #261612; border-left: 3px solid #ff6b5e;
                 padding: 0.65rem 0.9rem; border-radius: 6px; margin-top: 0.6rem; }}
    .verdict-tag {{ display:inline-block; background:{GROWW_GREEN_DARK}; color:#04140f;
                    padding:2px 9px; border-radius:20px; font-size:0.75rem;
                    font-weight:700; margin-right:8px; }}
    div[data-testid="stSidebar"] {{ background-color:#0b0e12; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="groww-header"><h1>Groww · HDFC Fund Assistant</h1></div>'
    '<div class="groww-sub">Ask factual questions about the 5 covered HDFC funds. '
    "Answers cite their source page and never give investment advice.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Covered funds")
    for fund in CORPUS:
        st.markdown(f"- **{fund['fund_name']}**  \n<small>{fund['fund_category']}</small>",
                    unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading corpus index…")
def build_assistant() -> RAGAssistant:
    embedder = ChunkEmbedder()
    store = ChromaVectorStore()
    retriever = Retriever(embedder, store)
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    generator = (
        MistralAnswerGenerator(api_key=api_key)
        if api_key
        else UnavailableAnswerGenerator()
    )
    return RAGAssistant(retriever=retriever, generator=generator)


api_key_present = bool(os.getenv("MISTRAL_API_KEY", "").strip())

if not api_key_present:
    st.info(
        "**Offline mode:** no `MISTRAL_API_KEY` found in `.env`, so answers use the "
        "offline fallback generator (deterministic, no network). Add your key to "
        f"`.env` and restart for Mistral-generated answers. Store: `{VECTOR_DB_DIR}` "
        "is the persisted vector database."
    )

st.caption("Example: “What is the expense ratio of the HDFC Large Cap Fund?”")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about an HDFC fund…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    assistant = build_assistant()
    with st.chat_message("assistant"):
        with st.spinner("Checking guard, retrieving sources, generating…"):
            answer = assistant.answer(prompt)

        if answer.is_refusal:
            st.markdown(
                f'<div class="refusal"><span class="verdict-tag">REFUSED · {answer.verdict}</span>'
                f"{answer.text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(answer.text)
            if answer.url:
                st.markdown(
                    f'<div class="citation">📄 <b>Source:</b> '
                    f'<a href="{answer.url}" target="_blank">{answer.url}</a></div>',
                    unsafe_allow_html=True,
                )
        st.session_state.messages.append({"role": "assistant", "content": answer.text})
