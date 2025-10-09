from typing import Any, Dict
import logging

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

logger = logging.getLogger("arag.graph.grade_documents")


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    logger.info("Checking document relevance to question")
    try:
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = False  # Initialize as False for RAG path
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade.lower() == "yes":
                logger.debug("Document relevant")
                filtered_docs.append(d)
            else:
                logger.debug("Document not relevant")
                web_search = True
                continue
        logger.info(f"Graded {len(documents)} documents, {len(filtered_docs)} relevant")
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    except Exception as e:
        logger.error(f"Error in grade_documents: {e}")
        return {"documents": [], "question": state["question"], "web_search": True}