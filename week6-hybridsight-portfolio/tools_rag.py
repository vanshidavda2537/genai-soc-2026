# tools_rag.py
"""
RAG indexing pipeline + the search_documents tool.

This reuses the exact chunk -> embed -> store pipeline from Week 2 (DocBuddy Pro),
pointed at the same persist_directory convention ("./chroma_store") so behaviour
stays consistent across weeks.
"""
import os
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Config (same values as Week 2 — keep them in sync if you tune one)
# ---------------------------------------------------------------------------
CHROMA_PATH   = "./chroma_store"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
EMBED_MODEL   = "all-MiniLM-L6-v2"

vectorstore      = None
embeddings_model = None


# ---------------------------------------------------------------------------
# INDEXING
# ---------------------------------------------------------------------------
def index_documents(pdf_paths: list, progress_callback=None) -> int:
    """Chunk + embed + store one or more PDFs into ChromaDB. Returns chunk count.

    progress_callback, if provided, is called as progress_callback(fraction, desc)
    at each stage so a caller (e.g. app.py's gr.Progress) can show real status
    instead of a single jump from "started" to "done". It's optional and
    defaults to None so this function still works exactly as before for any
    caller that doesn't pass one."""
    global vectorstore, embeddings_model

    def _tick(fraction, desc):
        if progress_callback is not None:
            progress_callback(fraction, desc)

    all_texts     = []
    all_metadatas = []

    _tick(0.10, "Reading PDF(s)...")
    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path}")
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = splitter.split_text(text)

            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": Path(pdf_path).name,
                    "page":   page_num + 1,
                })

    if not all_texts:
        print("No extractable text found in the uploaded PDF(s).")
        return 0

    print(f"Total chunks: {len(all_texts)}")
    _tick(0.40, f"Split into {len(all_texts)} chunks...")

    if embeddings_model is None:
        _tick(0.55, "Loading embedding model...")
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    _tick(0.70, f"Embedding {len(all_texts)} chunks...")
    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings_model,
        metadatas=all_metadatas,
        persist_directory=CHROMA_PATH,
    )

    _tick(1.0, "Done!")
    print(f"Done. ChromaDB saved at {CHROMA_PATH}")
    return len(all_texts)


def load_existing_store() -> int:
    """Loads a previously persisted ChromaDB store on app startup, if one exists."""
    global vectorstore, embeddings_model

    if not Path(CHROMA_PATH).exists():
        return 0

    print("Found existing ChromaDB, loading it...")
    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model,
    )
    count = vectorstore._collection.count()
    print(f"Loaded {count} chunks from disk")
    return count


# ---------------------------------------------------------------------------
# TOOL — wraps the retriever for the agent
# ---------------------------------------------------------------------------
@tool
def search_documents(query: str) -> str:
    """Searches the user's uploaded PDF documents for information relevant to the
    query. Use this tool whenever the question could be answered from a document
    the user has uploaded to this app — document-specific facts, definitions, or
    any content they explicitly asked you to read. Returns the most relevant
    chunks together with their source filename and page number so answers can be
    cited. If no PDFs have been indexed yet, this tool says so explicitly instead
    of guessing — in that case, tell the user to upload a PDF first."""
    global vectorstore

    if vectorstore is None:
        return (
            "No documents have been indexed yet. Ask the user to upload a PDF "
            "using the file uploader and click 'Index Documents', then try this "
            "search again."
        )

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
    except Exception as e:
        return f"Document search failed due to an internal error: {e}"

    if not docs:
        return "No relevant content was found in the indexed documents for that query."

    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)