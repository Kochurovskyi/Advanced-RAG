from typing import List, TypedDict, Optional
from langchain.schema import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question
        generation: LLM generation (optional)
        web_search: Whether to add web search
        documents: List of retrieved documents
    """

    question: str
    generation: Optional[str]
    web_search: bool
    documents: List[Document]