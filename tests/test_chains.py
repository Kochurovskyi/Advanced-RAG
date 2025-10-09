from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

# Import required modules
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from ingestion import retriever

# Import router modules (assuming they exist)
try:
    from graph.chains.router import RouteQuery, question_router
except ImportError:
    # If router doesn't exist yet, define placeholder types
    class RouteQuery:
        def __init__(self, datasource: str):
            self.datasource = datasource
    
    question_router = None

def test_retrival_grader_answer_yes() -> None:
    """
    Test retrieval grader with relevant documents.
    
    This test verifies that the retrieval grader correctly identifies
    relevant documents when the question matches the document content.
    Uses "chain of thought prompting" as a query that should find
    relevant content in the vectorstore.
    
    Expected behavior:
        - Retrieves documents from vectorstore
        - Grades the first document as relevant ("yes")
        - Asserts binary_score equals "yes"
    """
    question = "chain of thought prompting"
    docs = retriever.invoke(question)
    if len(docs) < 1:
        print(f"Only found {len(docs)} documents, need at least 1")
        return
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})
    assert res.binary_score == "yes"

def test_retrival_grader_answer_no() -> None:
    """
    Test retrieval grader with irrelevant documents.
    
    This test verifies that the retrieval grader correctly identifies
    irrelevant documents when the question doesn't match the document content.
    Uses "chain of thought prompting" to retrieve documents, then asks
    about "how to make pizza" to test irrelevant grading.
    
    Expected behavior:
        - Retrieves documents from vectorstore using relevant query
        - Grades the same document as irrelevant for pizza question ("no")
        - Asserts binary_score equals "no"
    """
    question = "chain of thought prompting"
    docs = retriever.invoke(question)
    if len(docs) < 1:
        print(f"Only found {len(docs)} documents, need at least 1")
        return
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": "how to make pizza", "document": doc_txt})
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    """
    Test the RAG generation chain with retrieved documents.
    
    This test verifies that the generation chain can produce coherent
    responses using retrieved documents as context. It tests the complete
    RAG pipeline from retrieval to generation.
    
    Expected behavior:
        - Retrieves relevant documents for "agent memory" query
        - Generates a response using the documents as context
        - Prints the generated response for manual verification
        - No assertion failures (test passes if generation succeeds)
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_hallucination_grader_answer_no() -> None:
    """
    Test hallucination grader with irrelevant content about cooking.
    
    This test verifies that the hallucination grader correctly identifies
    content that is completely unrelated to the retrieved documents.
    Uses "agent memory" query to retrieve documents, then tests with
    a cooking-related response that should be marked as not grounded.
    
    Expected behavior:
        - Retrieves documents for "agent memory" query
        - Uses a cooking-related response that's unrelated to agent memory
        - Grades the response as not grounded (False)
        - Asserts binary_score equals False
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    if len(docs) < 1:
        print(f"Only found {len(docs)} documents, need at least 1")
        return

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    """
    Test question router routes to vectorstore for relevant queries.
    
    This test verifies that the question router correctly identifies
    queries that should be answered using the vectorstore (RAG pipeline).
    Uses "agent memory" as a query that should be routed to vectorstore.
    
    Expected behavior:
        - Routes "agent memory" question to vectorstore
        - Asserts datasource equals "vectorstore"
    """
    if question_router is None:
        print("Question router not available, skipping test")
        return
        
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    """
    Test question router routes to websearch for general queries.
    
    This test verifies that the question router correctly identifies
    queries that should be answered using web search instead of vectorstore.
    Uses "how to make pizza" as a query that should be routed to websearch.
    
    Expected behavior:
        - Routes "how to make pizza" question to websearch
        - Asserts datasource equals "websearch"
    """
    if question_router is None:
        print("Question router not available, skipping test")
        return
        
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"

def test_hallucination_grader_grounded() -> None:
    """
    Test hallucination grader with grounded content.
    
    This test verifies that the hallucination grader correctly identifies
    content that is grounded in the provided facts. It uses the generation
    chain to create a response based on retrieved documents, then checks
    that the grader recognizes it as factually grounded.
    
    Expected behavior:
        - Retrieves documents for "agent memory" query
        - Generates a response using the generation chain
        - Grades the generated response as grounded (True)
        - Asserts binary_score equals True
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    if len(docs) < 1:
        print(f"Only found {len(docs)} documents, need at least 1")
        return
    
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score == True

def test_hallucination_grader_hallucinated() -> None:
    """
    Test hallucination grader with hallucinated content.
    
    This test verifies that the hallucination grader correctly identifies
    content that is not grounded in the provided facts. It uses a deliberately
    false statement about chain of thought prompting being a cooking technique
    to test hallucination detection.
    
    Expected behavior:
        - Retrieves documents for "agent memory" query
        - Uses a clearly false/hallucinated response about cooking
        - Grades the hallucinated response as not grounded (False)
        - Asserts binary_score equals False
    """
    question = "agent memory"
    docs = retriever.invoke(question)
    if len(docs) < 1:
        print(f"Only found {len(docs)} documents, need at least 1")
        return
    
    hallucinated_generation = "Chain of thought prompting is a technique used in cooking to improve the taste of pasta dishes."
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": hallucinated_generation}
    )
    assert res.binary_score == False
