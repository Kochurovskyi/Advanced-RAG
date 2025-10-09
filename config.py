"""
Configuration settings for the ARAG application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))

# Web Search Configuration
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Graph Configuration
GRAPH_ENTRY_POINT = "ROUTE"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Constants
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
