# Advanced RAG (Retrieval-Augmented Generation)

A sophisticated LangGraph-based RAG system that intelligently routes questions between vector store retrieval and web search, with advanced hallucination detection and document grading capabilities.

![Advanced RAG Workflow](graph.png)

## 🚀 Features

- **Intelligent Question Routing**: Automatically determines whether to use vector store (RAG) or web search based on question content
- **Advanced Document Grading**: Evaluates retrieved documents for relevance before generation
- **Hallucination Detection**: Validates generated responses against source documents
- **Hybrid Search Strategy**: Combines vector store retrieval with web search for comprehensive coverage
- **Production-Ready Logging**: Comprehensive logging throughout the workflow
- **Comprehensive Testing**: Full test coverage with unit and integration tests
- **Type Safety**: Full type annotations and validation using TypedDict

## 🏗️ Architecture

The system uses LangGraph to create a stateful workflow with the following components:

### Workflow Overview

1. **Question Routing**: Routes incoming questions to either vector store or web search
2. **Document Retrieval**: Fetches relevant documents from vector store (if applicable)
3. **Document Grading**: Evaluates document relevance to the question
4. **Web Search Fallback**: Performs web search if no relevant documents are found
5. **Answer Generation**: Generates responses using retrieved context
6. **Hallucination Detection**: Validates that generated answers are grounded in source documents
7. **Retry Mechanism**: Retries generation if hallucination is detected

### Key Components

- **StateGraph**: Manages the workflow state and transitions
- **Node Functions**: Individual processing steps (retrieve, grade, generate, etc.)
- **Conditional Edges**: Dynamic routing based on document relevance and generation quality
- **Chains**: LangChain components for specific tasks (generation, grading, routing)

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key (for LLM)
- Tavily Search API key (for web search)
- Vector store with embedded documents

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Advanced-RAG.git
   cd Advanced-RAG
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   # Required API Keys
   GOOGLE_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # Optional Configuration
   MODEL_NAME=gemini-2.0-flash-lite
   WEB_SEARCH_MAX_RESULTS=3
   LOG_LEVEL=INFO
   ```

## 🚀 Usage

### Basic Usage

```python
from graph.graph import app

# Ask a question
result = app.invoke({"question": "What is chain of thought prompting?"})
print(result["generation"])
```

### Running the Main Application

```bash
python main.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_chains.py -v
pytest tests/test_integration.py -v
```

## 📁 Project Structure

```
Advanced-RAG/
├── graph/
│   ├── __init__.py
│   ├── graph.py              # Main LangGraph workflow definition
│   ├── state.py              # GraphState TypedDict definition
│   ├── logging_config.py     # Logging configuration
│   ├── chains/               # LangChain components
│   │   ├── __init__.py
│   │   ├── generation.py     # RAG generation chain
│   │   ├── retrieval_grader.py  # Document relevance grader
│   │   ├── hallucination_grader.py  # Hallucination detector
│   │   └── router.py         # Question routing logic
│   └── nodes/                # LangGraph node functions
│       ├── __init__.py
│       ├── retrieve.py       # Document retrieval
│       ├── grade_documents.py # Document grading
│       ├── generate.py       # Answer generation
│       └── web_search.py     # Web search functionality
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_chains.py        # Unit tests for chains
│   └── test_integration.py   # Integration tests
├── ingestion/                # Document ingestion (external)
├── config.py                 # Configuration management
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── graph.mmd                # Mermaid diagram source
├── graph.png                # Workflow visualization
└── README.md                # This file
```

## 🔧 Configuration

The system is highly configurable through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Google Gemini API key |
| `TAVILY_API_KEY` | Required | Tavily Search API key |
| `MODEL_NAME` | `gemini-2.0-flash-lite` | LLM model to use |
| `WEB_SEARCH_MAX_RESULTS` | `3` | Maximum web search results |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_RETRIES` | `3` | Maximum generation retries |

## 🧪 Testing

The project includes comprehensive testing:

### Test Coverage
- **Unit Tests**: Individual chain components
- **Integration Tests**: End-to-end workflow testing
- **Error Handling**: Edge case and error scenario testing

### Running Tests
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=graph --cov-report=html

# Run specific test categories
pytest tests/test_chains.py -v      # Chain unit tests
pytest tests/test_integration.py -v # Integration tests
```

## 📊 Performance

- **Response Time**: ~2-5 seconds for typical queries
- **Accuracy**: High relevance through document grading
- **Reliability**: Robust error handling and retry mechanisms
- **Scalability**: Designed for production deployment

## 🔍 Workflow Details

### 1. Question Routing
The system first determines whether a question should be answered using:
- **Vector Store (RAG)**: For questions about AI, machine learning, prompt engineering, etc.
- **Web Search**: For general knowledge, current events, or topics not in the vector store

### 2. Document Processing
When using RAG:
- Retrieves relevant documents from vector store
- Grades each document for relevance to the question
- Triggers web search if no relevant documents are found

### 3. Answer Generation
- Generates responses using retrieved context
- Validates that answers are grounded in source documents
- Retries generation if hallucination is detected

### 4. Quality Assurance
- Hallucination detection ensures factual accuracy
- Document grading maintains relevance
- Comprehensive logging for debugging and monitoring

## 🛡️ Error Handling

The system includes robust error handling:
- Graceful fallbacks for API failures
- Retry mechanisms for transient errors
- Comprehensive logging for debugging
- State validation to prevent crashes

## 📈 Monitoring

Built-in logging provides insights into:
- Question routing decisions
- Document retrieval and grading results
- Generation quality and retry attempts
- Performance metrics and timing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) for the workflow framework
- [LangChain](https://python.langchain.com/) for the LLM integration
- [Tavily](https://tavily.com/) for web search capabilities
- [Google Gemini](https://ai.google.dev/) for the language model

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

---

**Built with ❤️ using LangGraph and LangChain**