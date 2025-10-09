from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
import pprint

from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.router import question_router
from graph.const import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, retrieve, web_search, grade_documents
from graph.state import GraphState
from graph.logging_config import logger

load_dotenv()


def validate_state(state: GraphState) -> bool:
    """
    Validate that the state contains required fields.
    
    Args:
        state: The current graph state
        
    Returns:
        bool: True if state is valid, False otherwise
    """
    required_fields = ["question"]
    for field in required_fields:
        if field not in state:
            logger.warning(f"Missing required field '{field}' in state")
            return False
    return True


def decide_to_generate(state):
    logger.info("Assessing graded documents")
    web_search_needed = state.get("web_search", False)
    if web_search_needed: 
        logger.info("Decision: Not all documents are relevant to question, include web search")
        return WEBSEARCH
    else:
        logger.info("Decision: Generate response")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Check if generation is grounded in documents.
    Args: state: GraphState containing documents and generation
    Returns: "useful" if grounded, "not supported" if not grounded
    """
    logger.info("Checking hallucinations")
    try:
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        
        if not documents or not generation:
            logger.warning("Missing documents or generation")
            return "not supported"
            
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        if score.binary_score:
            logger.info("Generation is grounded in documents")
            return "useful"
        else:
            logger.info("Generation is not grounded in documents (hallucinations), re-try")
            return "not supported"
    except Exception as e:
        logger.error(f"Error in hallucination check: {e}")
        return "not supported"

def route_question(state: GraphState) -> dict:
    """
    Route question to appropriate processing path.
    
    Uses the question router to determine whether to use vectorstore (RAG)
    or web search based on the question content.
    
    Args:
        state: GraphState containing question
        
    Returns:
        Dictionary with routing decision for state update
    """
    logger.info("Routing question")
    question = state.get("question", "")
    
    if not question:
        logger.warning("Empty or missing question, defaulting to RAG")
        return {"route": RETRIEVE}
    
    try:
        result = question_router.invoke({"question": question})
        if result.datasource == "vectorstore":
            logger.info("Route question to RAG (vectorstore)")
            return {"route": RETRIEVE}
        else:
            logger.info("Route question to web search")
            return {"route": WEBSEARCH}
    except Exception as e:
        logger.error(f"Router error: {e}, defaulting to RAG")
        return {"route": RETRIEVE}


def route_question_conditional(state: GraphState) -> str:
    """
    Conditional routing function for LangGraph edges.
    
    This function determines the next node based on the routing decision.
    """
    logger.info("Routing question (conditional)")
    question = state.get("question", "")
    
    if not question:
        logger.warning("Empty or missing question, defaulting to RAG")
        return RETRIEVE
    
    try:
        result = question_router.invoke({"question": question})
        if result.datasource == "vectorstore":
            logger.info("Route question to RAG (vectorstore)")
            return RETRIEVE
        else:
            logger.info("Route question to web search")
            return WEBSEARCH
    except Exception as e:
        logger.error(f"Router error: {e}, defaulting to RAG")
        return RETRIEVE


workflow = StateGraph(GraphState)
workflow.add_node("ROUTE", route_question)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Set router as entry point
workflow.set_entry_point("ROUTE")

# Add conditional edges from router
workflow.add_conditional_edges(
    "ROUTE", 
    route_question_conditional,
    {
        RETRIEVE: RETRIEVE,
        WEBSEARCH: WEBSEARCH
    }
)

# Add edges for RAG pipeline
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate,{WEBSEARCH: WEBSEARCH,GENERATE: GENERATE})
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question,{"not supported": GENERATE,"useful": END})
workflow.add_edge(WEBSEARCH, GENERATE)
app = workflow.compile()


# Test the graph with a sample question
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("TESTING RAG GRAPH")
    logger.info("="*50)
    # Generate Mermaid diagram file using app.get_graph().draw_mermaid_
    mermaid_content = app.get_graph().draw_mermaid()
    with open("graph.mmd", "w") as f: f.write(mermaid_content)
    logger.info("Mermaid file created as graph.mmd - use https://mermaid.live/ to generate PNG")
    logger.info("Graph compiled successfully!")
    # Test with a sample question
    test_question = "What is chain of thought prompting?"
    logger.info(f"Testing with question: {test_question}")
    try:
        result = app.invoke({"question": test_question})
        logger.info(f"Final result: {result}")
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        import traceback
        traceback.print_exc()