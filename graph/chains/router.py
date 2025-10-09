from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()
MODEL_NAME = "gemini-2.0-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(...,
    description="Given a user question choose to route it to web search or a vectorstore.",)


llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to:
- Agents and agent memory
- Prompt engineering techniques (including chain of thought prompting, few-shot prompting, etc.)
- Adversarial attacks in AI
- AI frameworks/libraries like LangGraph, LangChain
- Machine learning concepts and techniques

Use the vectorstore for questions on these topics. For all other questions (like cooking, weather, general knowledge), use web search."""
route_prompt = ChatPromptTemplate.from_messages(
    [("system", system),("human", "{question}"),])

question_router = route_prompt | structured_llm_router


def main():
    """
    Test the question router with sample questions.
    This function demonstrates the router's ability to classify questions
    and route them to either vectorstore or websearch based on content.
    """
    print("=" * 50)
    print("TESTING QUESTION ROUTER")
    print("=" * 50)
    
    # Test cases for different types of questions
    test_questions = [
        "What is chain of thought prompting?",
        "How does agent memory work?",
        "What are adversarial attacks in AI?",
        "How to make pizza?",
        "What's the weather today?",
        "Explain prompt engineering techniques",
        "How to fix a car engine?",
        "What is LangGraph?"]
    
    print(f"Testing {len(test_questions)} questions...\n")
    for i, question in enumerate(test_questions, 1):
        try:
            print(f"{i}. Question: {question}")
            result = question_router.invoke({"question": question})
            print(f"   Route: {result.datasource}")
            print(f"   Reasoning: {'Vectorstore' if result.datasource == 'vectorstore' else 'Web Search'}")
            print()
        except Exception as e: print(f"   Error: {e}")
    print("=" * 50)
    print("ROUTER TEST COMPLETE")
    print("=" * 50)


if __name__ == '__main__':
    main()