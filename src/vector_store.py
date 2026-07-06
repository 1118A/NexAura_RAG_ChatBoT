import os
import chromadb
from config import (
    CHROMA_MODE,
    CHROMA_PATH,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SSL,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    CHROMA_API_KEY,
    COLLECTION_NAME,
)

_client = None


def _build_client():
    """Create and return a fresh ChromaDB client based on config."""
    if CHROMA_MODE == "remote":
        if CHROMA_HOST == "api.trychroma.com":
            return chromadb.CloudClient(
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
                api_key=CHROMA_API_KEY,
            )
        else:
            headers = {}
            if CHROMA_API_KEY:
                headers["Authorization"] = f"Bearer {CHROMA_API_KEY}"
                headers["X-Chroma-Token"] = CHROMA_API_KEY

            return chromadb.HttpClient(
                host=CHROMA_HOST,
                port=int(CHROMA_PORT) if CHROMA_PORT else 8000,
                ssl=CHROMA_SSL,
                tenant=CHROMA_TENANT or "default_tenant",
                database=CHROMA_DATABASE or "default_database",
                headers=headers if headers else None,
            )
    else:
        # Use an absolute path so the DB location is deterministic regardless
        # of the working directory — prevents accidental relative-path drift.
        abs_path = os.path.abspath(CHROMA_PATH)
        os.makedirs(abs_path, exist_ok=True)
        return chromadb.PersistentClient(path=abs_path)


def get_client():
    """
    Return a cached ChromaDB client, rebuilding it if the connection is stale.

    Security guardrails:
    - Uses absolute path for local DB to prevent path traversal via cwd changes.
    - Resets stale / broken connections automatically instead of silently failing.
    """
    global _client
    if _client is None:
        _client = _build_client()
        return _client

    # Heartbeat check — reset stale remote connections
    try:
        _client.heartbeat()
    except Exception:
        _client = _build_client()

    return _client


def get_collection():
    """Get or create the configured ChromaDB collection."""
    client = get_client()
    return client.get_or_create_collection(COLLECTION_NAME)


def store_chunks(chunks: list[str], embeddings) -> object:
    """
    Store document chunks and their embeddings in a fresh collection.

    Security guardrails:
    - Validates inputs are non-empty before touching the DB.
    - Deletes and recreates the collection to avoid stale data from previous uploads.
    """
    if not chunks:
        raise ValueError("No chunks provided to store.")
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings provided to store.")

    client = get_client()

    # Delete old collection to prevent stale data accumulation
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet — safe to ignore

    collection = client.create_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    vectors = [emb.tolist() for emb in embeddings]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vectors,
    )

    return collection