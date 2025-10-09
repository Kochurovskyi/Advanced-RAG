from typing import Any, Dict
import logging

from graph.state import GraphState
from ingestion import retriever

logger = logging.getLogger("arag.graph.retrieve")


def retrieve(state: GraphState) -> Dict[str, Any]:
    logger.info("Retrieving documents")
    try:
        question = state.get("question", "")
        if not question:
            logger.warning("Empty question in retrieve")
            return {"documents": [], "question": ""}
        documents = retriever.invoke(question)
        logger.info(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "question": question}
    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        return {"documents": [], "question": state.get("question", "")}