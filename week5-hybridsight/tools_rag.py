from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding_model,
    collection_name="lecture_notes",
)


def index_documents(file_paths: list) -> int:
    """
    Read one or more PDFs, split into chunks, and store in ChromaDB.
    Returns the total number of chunks indexed.
    """
    total_chunks = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        vectorstore.add_documents(chunks)
        total_chunks += len(chunks)

    return total_chunks


@tool
def search_documents(query: str) -> str:
    """
    Search the user's uploaded documents for information relevant to
    the query.

    Use this tool when the user asks about:
    - uploaded PDFs
    - lecture notes
    - "my notes"
    - "the document"

    Do NOT use it for:
    - current news
    - general knowledge
    """

    if vectorstore._collection.count() == 0:
        return "No documents have been uploaded yet."

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    chunks = retriever.invoke(query)

    if not chunks:
        return "No relevant information found."

    result = []

    for chunk in chunks:
        page = chunk.metadata.get("page", "?")
        result.append(
            f"[Page {page}] {chunk.page_content}"
        )

    return "\n\n".join(result)