from src.embedding import embed_query
from src.vector_store import get_collection

def retrieve(query, top_k=3):

    collection = get_collection()

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results["documents"][0]