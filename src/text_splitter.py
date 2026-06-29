from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )

    return splitter.split_text(text)


