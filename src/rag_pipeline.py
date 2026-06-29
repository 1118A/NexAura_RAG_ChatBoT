from src.pdf_load import load_pdf
from src.text_splitter import split_text
from src.embedding import create_embeddings
from src.vector_store import store_chunks
from src.retriever import retrieve
from src.llm import generate_answer

def process_pdf(pdf_path):

    text = load_pdf(pdf_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    store_chunks(chunks, embeddings)

def ask_question(question, provider="Gemini", model_name="gemini-2.5-flash", api_key=None):

    context = retrieve(question)

    context = "\n\n".join(context)

    answer = generate_answer(context, question, provider=provider, model_name=model_name, api_key=api_key)

    return answer