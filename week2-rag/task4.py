import os
import gradio as gr

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Groq Setup

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

CHROMA_PATH = "./chroma_store"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = None


# Index Documents

def index_documents(files):

    global vector_store

    all_documents = []

    for file in files:

        loader = PyPDFLoader(file.name)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(
                file.name
            )

        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        all_documents
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    return (
        f"Indexed {len(files)} document(s) "
        f"and created {len(chunks)} chunks."
    )



# Ask Question

def ask_question(question, history):

    global vector_store

    if vector_store is None:

      history.append(
        {
            "role": "user",
            "content": question
        }
    )

      history.append(
        {
            "role": "assistant",
            "content": "Please index documents first."
        }
    )

      return history, ""

    docs = vector_store.similarity_search(
        question,
        k=4
    )

    context = ""

    retrieved_context = ""

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context += (
            f"\nSource: {source}, "
            f"Page: {page + 1}\n"
        )

        context += doc.page_content
        context += "\n\n"

        retrieved_context += (
            f"\n\nFILE: {source}"
            f"\nPAGE: {page + 1}"
            f"\n{'-'*50}\n"
        )

        retrieved_context += doc.page_content

    prompt = f"""
You are a grounded QA assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply exactly:

I don't have that information in the uploaded documents.

Always cite sources.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    history.append(
    {
        "role": "user",
        "content": question
    }
)

    history.append(
    {
        "role": "assistant",
        "content": answer
    }
)

    return history, retrieved_context


# Gradio UI

with gr.Blocks() as demo:

    gr.Markdown(
        "# PDF Grounded Q&A System"
    )

    with gr.Row():

        pdf_files = gr.File(
            file_count="multiple",
            file_types=[".pdf"],
            label="Upload PDFs"
        )

        index_button = gr.Button(
            "Index Documents"
        )

    status = gr.Textbox(
        label="Status"
    )

    chatbot = gr.Chatbot(
       
        label="Conversation"
    )

    question = gr.Textbox(
        label="Ask a Question"
    )

    retrieved_chunks = gr.Textbox(
        label="Retrieved Context"
    )

    history_state = gr.State([])

    index_button.click(
        fn=index_documents,
        inputs=[pdf_files],
        outputs=[status]
    )

    question.submit(
        fn=ask_question,
        inputs=[
            question,
            history_state
        ],
        outputs=[
            chatbot,
            retrieved_chunks
        ]
    )

demo.launch()