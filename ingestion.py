from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import os

load_dotenv()
MODEL_NAME = "gemini-2.0-flash-lite"
embd = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

# Set USER_AGENT to avoid detection
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
url = "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"

# -------------------------------------------- Retriever Creation --------------------------------
def create_retriever():
    """Create and return a Chroma retriever."""
    return Chroma(
        collection_name="rag-chroma", 
        persist_directory="./.chroma", 
        embedding_function=embd
    ).as_retriever(search_kwargs={"k": 3})

def create_vectorstore_from_urls(urls, chunk_size=250, chunk_overlap=0):
    """Create vectorstore from URLs with retry logic."""
    max_retries = 3
    docs = None
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to load: {urls}")
            docs = WebBaseLoader(urls).load()
            print(f"Successfully loaded {len(docs)} documents")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                print("All attempts failed!")
                raise e
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits, 
        collection_name="rag-chroma", 
        embedding=embd, 
        persist_directory="./.chroma"
    )
    print("Vectorstore created successfully!")
    return vectorstore

# Create retriever instance
retriever = create_retriever()

# -------------------------------------------- Retriever Usage/Testing --------------------------------
def test_retriever(retriever, test_queries=None):
    """Test the retriever with sample queries."""
    if test_queries is None:
        test_queries = [
            "What is chain of thought prompting?",
            "How does few-shot learning work?",
            "What are the benefits of instruction prompting?",
            "Explain self-consistency sampling"
        ]
    
    print("Retriever created:", retriever)
    print('\n' + '='*50)
    print("TESTING RETRIEVAL WITH SAMPLE QUERIES")
    print('='*50)

    for i, query in enumerate(test_queries, 1):
        print(f'\n--- Query {i}: {query} ---')
        try:
            # Retrieve top 3 most relevant documents
            docs = retriever.invoke(query)
            print(f"Found {len(docs)} relevant documents:")
            
            for j, doc in enumerate(docs[:3], 1):  # Show top 3 results
                print(f"\nResult {j}:")
                print(f"Content preview: {doc.page_content[:300]}...")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Title: {doc.metadata.get('title', 'Unknown')}")
                print("-" * 40)
                
        except Exception as e:
            print(f"Error retrieving for query '{query}': {e}")

    print('\n' + '='*50)
    print("RETRIEVAL TEST COMPLETED")
    print('='*50)

if __name__ == '__main__':
    print('-------------------------------------------- RAG System Setup --------------------------------')
    
    # Example: Create new vectorstore from URLs
    print("Creating vectorstore from URLs...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
    vectorstore = create_vectorstore_from_urls(urls, chunk_size=1000, chunk_overlap=100)
    new_retriever = create_retriever()
    
    # Test the retriever
    test_retriever(new_retriever)
