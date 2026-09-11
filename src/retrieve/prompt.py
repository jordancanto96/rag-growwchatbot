from src.config import CORPUS


def build_context_passages(hits: list) -> list[str]:
    passages = []
    for hit in hits:
        passages.append(
            "- {fund_category} | {fund_name}: {chunk_text}".format(
                fund_category=getattr(hit, "fund_category", ""),
                fund_name=getattr(hit, "fund_name", ""),
                chunk_text=getattr(hit, "chunk_text", ""),
            )
        )
    return passages


def build_system_prompt(question: str) -> str:
    return (
        "You are a helpful assistant that answers questions about HDFC mutual "
        "funds. Answer ONLY using the provided context, in plain English, in "
        "2-4 sentences. Cite the fund name and category. If the context does "
        "not contain the answer, say you don't know. Never invent facts and "
        "never give investment advice.\n\n"
        f"Question: {question}"
    )


def build_prompt(question: str, hits: list) -> str:
    passages = build_context_passages(hits)
    if not passages:
        return (
            "No matching fund information was found in the corpus. Please rephrase "
            "your question or ask about one of the covered HDFC funds."
        )
    context = "\n".join(passages)
    fund_names = ", ".join(dict.fromkeys(getattr(h, "fund_name", "") for h in hits))
    return (
        "Context from fund documents:\n"
        f"{context}\n\n"
        f"Covered funds: {fund_names}.\n\n"
        "Answer the following question in 2-4 sentences, in plain English, "
        "citing the fund name and category.\n"
        f"Question: {question}"
    )


def list_covered_funds() -> str:
    return "; ".join(f["fund_name"] for f in CORPUS)
