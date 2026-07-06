from src.embedding import embed_query
from src.vector_store import get_collection


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """
    Retrieve the top-k most relevant document chunks for a given query.

    Security guardrails:
    - Validates the query is non-empty before embedding.
    - Catches ChromaDB 'collection not found' and 'empty collection' errors
      and returns an empty list instead of raising a raw traceback.
    - Catches all other unexpected errors and raises a clean message.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        collection = get_collection()
        query_embedding = embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results.get("documents", [[]])[0]

    except Exception as exc:
        err = str(exc).lower()
        # Gracefully handle empty or missing collection
        if any(k in err for k in ("does not exist", "not found", "no documents", "empty")):
            return []
        # Re-raise a clean, non-revealing error for everything else
        raise RuntimeError(
            "Failed to query the document index. "
            "Please re-index the PDF and try again."
        ) from None