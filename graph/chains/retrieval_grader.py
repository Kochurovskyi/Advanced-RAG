from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ingestion import retriever

load_dotenv()
MODEL_NAME = "gemini-2.0-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),])

retrieval_grader = grade_prompt | structured_llm_grader


def main():
    """Test the retrieval grader with sample queries"""
    print("Testing Retrieval Grader...")
    
    # Test with relevant query
    question = "chain of thought prompting"
    docs = retriever.invoke(question)
    
    if len(docs) < 1:
        print(f"No documents found for query: {question}")
        return
    
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})
    print(f"Query: {question}")
    print(f"Relevant: {res.binary_score}")
    print(f"Test passed: {res.binary_score == 'yes'}")
    
    # Test with irrelevant query
    question2 = "how to make pizza"
    docs2 = retriever.invoke(question2)
    if len(docs2) > 0:
        doc_txt2 = docs2[0].page_content
        res2: GradeDocuments = retrieval_grader.invoke({"question": question2, "document": doc_txt2})
        print(f"Query: {question2}")
        print(f"Relevant: {res2.binary_score}")
        print(f"Test passed: {res2.binary_score == 'no'}")

if __name__ == '__main__': main()