from typing import Any, Dict
import logging
from graph.chains.generation import generation_chain
from graph.state import GraphState

logger = logging.getLogger("arag.graph.generate")


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response using the RAG generation chain with retrieved documents.
    
    This function takes a question and retrieved documents, then uses the generation
    chain to create a coherent response based on the provided context. It includes
    sanity checks to ensure the input data is valid before processing.
    
    Args:
        state (GraphState): The current graph state containing:
            - question (str): The user's question to generate a response for
            - documents (list[Document]): Retrieved documents to use as context
    
    Returns:
        Dict[str, Any]: Updated state containing:
            - documents (list[Document]): Original retrieved documents (preserved)
            - question (str): Original question (preserved)
            - generation (str): Generated response text
    
    Raises:
        ValueError: If required fields are missing or invalid
        RuntimeError: If generation chain fails to produce output
    
    Example:
        >>> state = {
        ...     "question": "What is machine learning?",
        ...     "documents": [Document(page_content="ML is...", metadata={})]
        ... }
        >>> result = generate(state)
        >>> print(result["generation"])
        Machine learning is a subset of artificial intelligence...
    
    Note:
        - Requires valid question and documents in the state
        - Uses the generation_chain from graph.chains.generation
        - Preserves original documents and question in output
    """
    logger.info("Generating response")   
    question = state["question"]
    documents = state["documents"]
    try:
        generation = generation_chain.invoke({"context": documents, "question": question})
        logger.info(f"Generated response length: {len(generation)} characters")
    except Exception as e:
        logger.error(f"Generation chain failed: {str(e)}")
        raise RuntimeError(f"Generation chain failed: {str(e)}")
    web_search = state.get("web_search", False)
    return {"documents": documents, "question": question, "generation": generation, "web_search": web_search}


if __name__ == "__main__":
    """Test the generate function when run directly"""
    from pprint import pprint
    
    try:
        from ingestion import retriever
        
        print("Testing generate function...")
        print("="*50)
        
        # Test with a real question and documents
        question = "What is chain of thought prompting?"
        print(f"Question: {question}")
        
        # Retrieve documents
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents")
        
        # Create state
        state = GraphState(question=question, documents=docs)
        
        # Generate response
        result = generate(state)
        
        print("\nGenerated Response:")
        print("-" * 30)
        print(result["generation"])
        print("-" * 30)
        print(f"Response length: {len(result['generation'])} characters")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()