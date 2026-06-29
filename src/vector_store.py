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
    COLLECTION_NAME
)

_client = None

def get_client():
    global _client
    if _client is None:
        if CHROMA_MODE == "remote":
            if CHROMA_HOST == "api.trychroma.com":
                # Use CloudClient automatically for Chroma Cloud
                _client = chromadb.CloudClient(
                    tenant=CHROMA_TENANT,
                    database=CHROMA_DATABASE,
                    api_key=CHROMA_API_KEY
                )
            else:
                # Use HttpClient for self-hosted instances
                headers = {}
                if CHROMA_API_KEY:
                    # Support standard bearer token and X-Chroma-Token headers
                    headers["Authorization"] = f"Bearer {CHROMA_API_KEY}"
                    headers["X-Chroma-Token"] = CHROMA_API_KEY
                
                _client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=int(CHROMA_PORT) if CHROMA_PORT else 8000,
                    ssl=CHROMA_SSL,
                    tenant=CHROMA_TENANT or "default_tenant",
                    database=CHROMA_DATABASE or "default_database",
                    headers=headers if headers else None
                )
        else:
            _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client

def get_collection():
    client = get_client()
    return client.get_or_create_collection(COLLECTION_NAME)


def store_chunks(chunks, embeddings):
    client = get_client()

    # Delete old collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create a new collection
    collection = client.create_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    vectors = [emb.tolist() for emb in embeddings]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vectors
    )

    return collection