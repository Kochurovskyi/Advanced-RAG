from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gemini-2.0-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
prompt = hub.pull("rlm/rag-prompt")
# Generation chain for RAG: combines prompt, LLM, and output parser
generation_chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    from ingestion import retriever
    from pprint import pprint
    question = "agent memory"
    docs = retriever.invoke(question)
    res = generation_chain.invoke({"context": docs, "question": question})
    pprint(res)