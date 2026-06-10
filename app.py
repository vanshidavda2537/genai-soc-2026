import os
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Config

CHROMA_PATH      = "./chroma_store"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
EMBED_MODEL      = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.1-8b-instant"

vectorstore      = None
embeddings_model = None


# INDEXING

def index_documents(pdf_paths: list) -> int:
    global vectorstore, embeddings_model

    all_texts     = []
    all_metadatas = []

    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path}")
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_text(text)

            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": Path(pdf_path).name,
                    "page":   page_num + 1
                })

    print(f"Total chunks: {len(all_texts)}")

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings_model,
        metadatas=all_metadatas,
        persist_directory=CHROMA_PATH
    )

    print(f"Done. ChromaDB saved at {CHROMA_PATH}")
    return len(all_texts)



# RETRIEVAL + GENERATION

def ask(question: str) -> tuple[str, str]:
    global vectorstore

    if vectorstore is None:
        return "No documents indexed yet. Upload some PDFs first.", ""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs      = retriever.invoke(question)

    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
- Only use information from the context below
- Always cite sources like [Source X: filename, Page Y]
- If the answer is not in the context, say exactly: "I don't have that information in the provided documents."
- Do NOT use your own training knowledge to fill gaps
- Be concise and factual"""

    user_prompt = f"""Context:
{context_text}

Question: {question}

Answer (with citations):"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    answer = response.content

    context_display = ""
    for i, doc in enumerate(docs):
        source  = doc.metadata.get("source", "Unknown")
        page    = doc.metadata.get("page", "?")
        preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        context_display += f"**Chunk {i+1}** &nbsp;|&nbsp; `{source}` &nbsp;|&nbsp; Page `{page}`\n\n{preview}\n\n---\n\n"

    return answer, context_display



# PERSISTENCE

def load_existing_store() -> int:
    global vectorstore, embeddings_model

    if not Path(CHROMA_PATH).exists():
        return 0

    print("Found existing ChromaDB, loading it...")
    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )
    count = vectorstore._collection.count()
    print(f"Loaded {count} chunks from disk")
    return count


# HANDLERS

def handle_upload(files):
    if not files:
        return "⚠️ Please upload at least one PDF."
    pdf_paths   = [f.name for f in files]
    chunk_count = index_documents(pdf_paths)
    return f"✅ {len(pdf_paths)} document(s) indexed — {chunk_count} chunks stored."


def chat_fn(question, history):
    if not question.strip():
        return history, "", ""
    answer, context = ask(question)
    history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]
    return history, "", context


# Gradio 6 theme — purple accent, light clean look

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#7c6ef7",
    button_primary_background_fill_hover="#6a5de0",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f0eeff",
    button_secondary_text_color="#7c6ef7",
    block_title_text_color="#1a1a1a",
    block_label_text_color="#888888",
    input_background_fill="#fafaf7",
    background_fill_primary="#f4f4f0",
    background_fill_secondary="#ffffff",
)


# UI

existing = load_existing_store()

with gr.Blocks(
    title="DocBuddy Pro",
    theme=theme,
) as demo:

    gr.Markdown("""
    # 📚 DocBuddy Pro
    **Multi-PDF Q&A** with source citations, grounded answers, and RAG pipeline visibility.
    """)

    with gr.Row():

        # ── Left: upload + controls 
        with gr.Column(scale=1):

            gr.Markdown("### 🗂 Knowledge Base")

            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDF Documents"
            )

            index_btn = gr.Button(
                "⟳  Index Documents",
                variant="primary",
                size="lg"
            )

            status_label = gr.Textbox(
                label="Status",
                value=f"{'✅ Loaded ' + str(existing) + ' existing chunks from disk.' if existing else 'No documents indexed yet.'}",
                interactive=False,
                lines=1
            )

            with gr.Accordion("ℹ️ How RAG works", open=False):
                gr.Markdown("""
**1. Chunking** — Each PDF page is split into 500-character overlapping chunks using `RecursiveCharacterTextSplitter`.

**2. Embedding** — Every chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2` (HuggingFace).

**3. Storage** — Vectors + metadata (filename, page) are stored in **ChromaDB** on disk.

**4. Retrieval** — Your question is embedded, then the top 5 closest chunks are fetched via cosine similarity.

**5. Generation** — Groq's LLM answers using *only* the retrieved chunks — preventing hallucination.
                """)

        # ── Right: chat 
        with gr.Column(scale=2):

            gr.Markdown("### 💬 Ask Your Documents")

            chatbot = gr.Chatbot(
                height=440,
                show_label=False,
                placeholder="Upload PDFs and index them on the left, then ask anything here.",
            )

            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False,
                    lines=1,
                    scale=5,
                    container=False
                )
                ask_btn = gr.Button("Ask →", variant="primary", scale=1)

            with gr.Accordion("🔎 Retrieved Context — see how RAG works under the hood", open=False):
                context_display = gr.Markdown(
                    value="Ask a question to see which chunks were retrieved here."
                )

    gr.Markdown(
        "<center><small>DocBuddy Pro · Powered by Groq · ChromaDB · LangChain · HuggingFace</small></center>"
    )

    # ── State & wiring 
    chat_history = gr.State([])

    index_btn.click(
        fn=handle_upload,
        inputs=[file_input],
        outputs=[status_label]
    )
    ask_btn.click(
        fn=chat_fn,
        inputs=[question_input, chat_history],
        outputs=[chatbot, question_input, context_display]
    )
    question_input.submit(
        fn=chat_fn,
        inputs=[question_input, chat_history],
        outputs=[chatbot, question_input, context_display]
    )


if __name__ == "__main__":
    demo.launch(share=False)