import os
from dotenv import load_dotenv

load_dotenv()

# Fallback API Keys from environment
Gemini_API_key = os.getenv("Gemini_API")
Groq_API_key = os.getenv("GROQ_API_KEY")
OpenAI_API_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag")

# Chroma Connection Settings (Local or Remote Server)
CHROMA_MODE = os.getenv("CHROMA_MODE", "local")  # Set to "remote" to use a hosted Chroma DB
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_SSL = os.getenv("CHROMA_SSL", "False").lower() in ("true", "1", "yes")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")  # API Token for Remote Chroma server authentication


# Provider and Model mappings
PROVIDERS = {
    "Gemini": {
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"]
    },
    "ChatGPT (OpenAI)": {
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o"]
    },
    "Groq": {
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
    }
}