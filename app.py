import os
from typing import Any, Dict, List, Optional
import streamlit as st
from dotenv import load_dotenv
from graph.graph import app as rag_app
load_dotenv()


def _env_flag(name: str) -> bool:
    return bool(os.getenv(name))

def _format_source(is_web: bool) -> str:
    return "Web search" if is_web else "RAG (Chroma)"


def _render_documents(documents: List[Any]) -> None:
    if not documents:
        st.info("No documents returned.")
        return

    st.subheader(f"📚 Documents ({len(documents)})")
    for i, doc in enumerate(documents, 1):
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("url") or "unknown"
        title = meta.get("title") or meta.get("name") or f"Document {i}"
        with st.expander(f"{i}. {title}"):
            st.caption(f"source: {source}")
            st.json(meta)
            content = getattr(doc, "page_content", "")
            st.write(content if content else "(empty)")


def _invoke_graph(question: str) -> Dict[str, Any]:
    # Keep input shape compatible with main.py usage.
    return rag_app.invoke({"question": question})


def main() -> None:
    st.set_page_config(page_title="Advanced RAG (LangGraph)", layout="wide")

    st.title("Advanced RAG (LangGraph)")
    st.write("Ask a question; the graph will route between **RAG** and **web search**.")

    with st.sidebar:
        st.markdown("### Configuration")
        st.write(f"GOOGLE_API_KEY set: `{_env_flag('GOOGLE_API_KEY') or _env_flag('GEMINI_API_KEY')}`")
        # st.write(f"TAVILY_API_KEY set: `{_env_flag('TAVILY_API_KEY')}`")

        st.markdown("### Examples")
        examples = [
            "What is short term memory in agents?",
            "What is super memory in agents?",
            "What is chain of thought prompting?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["question"] = ex

        show_debug = st.checkbox("Show debug output", value=False)
        show_docs = st.checkbox("Show documents", value=True)

    question = st.text_input(
        "Question",
        value=st.session_state.get("question", ""),
        placeholder="e.g. What is chain of thought prompting?",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a: run = st.button("Run", type="primary", use_container_width=True)
    # with col_b: t.caption("Tip: set `GOOGLE_API_KEY`/`GEMINI_API_KEY` and `TAVILY_API_KEY` in `.env` for full functionality.")

    if not run:
        return

    if not question.strip():
        st.warning("Enter a question.")
        return

    with st.spinner("Running graph..."):
        result = _invoke_graph(question.strip())

    st.subheader("Answer")
    st.markdown(result.get("generation", "(no generation)"))

    # st.subheader("Routing")
    st.write(f"Source: **{_format_source(bool(result.get('web_search')))}**")

    if show_docs:
        _render_documents(result.get("documents", []) or [])

    if show_debug:
        st.subheader("Raw state")
        st.json(result)


if __name__ == "__main__":
    main()