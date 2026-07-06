from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str) -> list[str]:
    """
    Split document text into overlapping chunks for embedding.

    Security guardrails:
    - Guards against empty or whitespace-only input to prevent
      downstream failures in the embedding and vector store steps.
    """
    if not text or not text.strip():
        raise ValueError("Cannot split empty text. The document appears to have no content.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError(
            "Text splitting produced no chunks. "
            "The document may be too short or contain only whitespace."
        )

    return chunks
