from typing import Any, Dict
import logging
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from graph.state import GraphState
from config import WEB_SEARCH_MAX_RESULTS

load_dotenv()
logger = logging.getLogger("arag.graph.web_search")

def _build_web_search_tool():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_tavily import TavilySearch
    except Exception:
        return None
    return TavilySearch(max_results=WEB_SEARCH_MAX_RESULTS, api_key=api_key)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform web search using Tavily Search API and return results as documents.
    
    This function searches the web for information related to the user's question
    and returns the results as LangChain Document objects. It can either append
    results to existing documents or create a new list of documents.
    
    Args:
        state (GraphState): The current graph state containing:
            - question (str): The user's question to search for
            - documents (list[Document], optional): Existing documents to append to
    
    Returns:
        Dict[str, Any]: Updated state containing:
            - documents (list[Document]): List of documents including web search results
            - question (str): The original question (preserved)
    
    Example:
        >>> state = {"question": "What is machine learning?", "documents": []}
        >>> result = web_search(state)
        >>> print(f"Found {len(result['documents'])} documents")
        Found 1 documents
    
    Note:
        - Uses Tavily Search API with max_results=3
        - Web search results are combined into a single Document
        - If no existing documents, creates new list with web results
        - If existing documents, appends web results to the list
    """
    logger.info("Performing web search")
    try:
        question = state["question"]
        documents = state.get("documents", None)

        web_search_tool = _build_web_search_tool()
        if web_search_tool is None:
            logger.warning("Tavily not configured; skipping web search")
            return {"documents": documents or [], "question": question, "web_search": True}
        
        tavily_results = web_search_tool.invoke({"query": question})["results"]
        joined_tavily_result = "\n".join([tavily_result["content"] for tavily_result in tavily_results])
        web_results = Document(page_content=joined_tavily_result)
        
        if documents is not None: 
            documents.append(web_results)
        else: 
            documents = [web_results]
            
        logger.info(f"Web search completed, found {len(tavily_results)} results")
        return {"documents": documents, "question": question, "web_search": True}
    except Exception as e:
        logger.error(f"Error in web_search: {e}")
        return {"documents": [], "question": state["question"], "web_search": True}


if __name__ == "__main__":
    res = web_search(state={"question": "agent memory", "documents": None})
    print(res)