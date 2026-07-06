from src.pdf_load import load_pdf
from src.text_splitter import split_text
from src.embedding import create_embeddings
from src.vector_store import store_chunks
from src.retriever import retrieve
from src.llm import generate_answer


def process_pdf(pdf_path: str) -> None:
    """
    Full ingestion pipeline: load → split → embed → store.

    Security guardrails:
    - Each step validates its output before passing to the next.
    - Errors are caught and re-raised with user-friendly messages
      that do NOT expose internal paths or stack frames.
    """
    try:
        text = load_pdf(pdf_path)
    except (FileNotFoundError, ValueError, RuntimeError):
        raise  # Already user-friendly; propagate as-is
    except Exception:
        raise RuntimeError("An unexpected error occurred while reading the PDF.")

    try:
        chunks = split_text(text)
    except ValueError:
        raise  # Already user-friendly
    except Exception:
        raise RuntimeError("An unexpected error occurred while splitting the document.")

    if not chunks:
        raise ValueError("The document produced no processable text chunks.")

    try:
        embeddings = create_embeddings(chunks)
    except Exception:
        raise RuntimeError(
            "Failed to generate embeddings. "
            "Please check your network connection and try again."
        )

    try:
        store_chunks(chunks, embeddings)
    except Exception:
        raise RuntimeError(
            "Failed to store the document in the vector database. "
            "Please check your ChromaDB configuration and try again."
        )


def ask_question(
    question: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.5-flash",
    api_key: str = None,
) -> str:
    """
    Retrieve relevant context from vector DB and generate an LLM answer.

    Security guardrails:
    - Validates question is non-empty.
    - Errors are user-friendly and do not expose internals.
    """
    question = (question or "").strip()
    if not question:
        return "⚠️ Please enter a valid question."

    try:
        context_chunks = retrieve(question)
    except Exception:
        raise RuntimeError(
            "Failed to retrieve context from the document. "
            "Please ensure the PDF has been indexed before asking questions."
        )

    context = "\n\n".join(context_chunks) if context_chunks else ""

    try:
        answer = generate_answer(
            context, question,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
        )
    except (ValueError, RuntimeError):
        raise  # Already user-friendly
    except Exception:
        raise RuntimeError(
            "An unexpected error occurred while generating the answer. "
            "Please try again."
        )

    return answer