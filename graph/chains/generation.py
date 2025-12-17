import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODEL_NAME = "gemini-2.0-flash-lite"


def _build_prompt() -> ChatPromptTemplate:
    # Simple, local RAG prompt (avoids langchain hub API drift).
    return ChatPromptTemplate.from_template(
        "You are a helpful assistant.\n"
        "Use the following context to answer the question.\n"
        "If the context is empty or insufficient, say so and answer from general knowledge.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )


def _build_llm():
    """Build LLM lazily to avoid import-time failures without API keys."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=api_key)


def _fallback_answer(_inputs: dict) -> str:
    return (
        "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY to enable generation.\n"
        "Returned without calling an LLM."
    )


def _get_generation_chain():
    prompt = _build_prompt()
    llm = _build_llm()
    if llm is None:
        # Keep interface consistent with LCEL: callable that returns string.
        return _fallback_answer
    return prompt | llm | StrOutputParser()


# Generation chain for RAG (callable or LCEL runnable).
generation_chain = _get_generation_chain()


if __name__ == "__main__":
    from ingestion import retriever
    from pprint import pprint
    question = "agent memory"
    docs = retriever.invoke(question)
    # generation_chain may be a runnable or a simple callable.
    if hasattr(generation_chain, "invoke"):
        res = generation_chain.invoke({"context": docs, "question": question})
    else:
        res = generation_chain({"context": docs, "question": question})
    pprint(res)