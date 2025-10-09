from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gemini-2.0-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeHallucinations)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")])

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader


def main():
    """Test the hallucination grader with sample queries"""
    print("Testing Hallucination Grader...")
    
    try:
        from ingestion import retriever
        from graph.chains.generation import generation_chain
        
        # Test with relevant query and generation
        question = "chain of thought prompting"
        docs = retriever.invoke(question)
        
        if len(docs) < 1:
            print(f"No documents found for query: {question}")
            return
        
        # Generate a response using the generation chain
        generation = generation_chain.invoke({"context": docs, "question": question})
        print(f"Generated response: {generation[:100]}...")
        
        # Test hallucination grader
        res: GradeHallucinations = hallucination_grader.invoke({
            "documents": docs, 
            "generation": generation
        })
        print(f"Query: {question}")
        print(f"Grounded in facts: {res.binary_score}")
        print(f"Test passed: {res.binary_score == True}")
        
        # Test with hallucinated content
        print("\n--- Testing with hallucinated content ---")
        hallucinated_generation = "Chain of thought prompting is a technique used in cooking to improve the taste of pasta dishes."
        res2: GradeHallucinations = hallucination_grader.invoke({
            "documents": docs, 
            "generation": hallucinated_generation
        })
        print(f"Hallucinated response: {hallucinated_generation}")
        print(f"Grounded in facts: {res2.binary_score}")
        print(f"Test passed: {res2.binary_score == False}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()