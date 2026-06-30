import gradio as gr
from agent import chat
from tools_rag import index_documents


def upload_pdf(files):
    """
    Index uploaded PDF(s) into ChromaDB.
    """
    if files is None:
        return "No PDF uploaded."

    pdf_paths = [file.name for file in files]
    total = index_documents(pdf_paths)

    return f"✅ Indexed {len(pdf_paths)} PDF(s) — {total} chunks stored."


def chat_fn(message, image, history):
    """
    Sends the user's question + optional image to the LangGraph agent.
    """
    if image:
        safe_image_path = image.replace("\\", "/")
        full_message = f"{message} [IMAGE_PATH: {safe_image_path}]"
    else:
        full_message = message

    answer = chat(full_message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history


with gr.Blocks(title="HybridSight") as demo:

    gr.Markdown(
        "# 🤖 HybridSight\n"
        "RAG + Web Search + Vision Agent"
    )

    pdf_upload = gr.File(
        label="Upload PDF(s)",
        file_count="multiple",
        file_types=[".pdf"]
    )

    upload_status = gr.Textbox(
        label="Upload Status",
        interactive=False
    )

    upload_button = gr.Button("Index PDFs")

    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_upload,
        outputs=upload_status
    )

    # Image Upload
    image = gr.Image(
        type="filepath",
        label="Upload Image"
    )

    chatbot = gr.Chatbot(
    height=500 
)

    message = gr.Textbox(placeholder="Ask anything...")

    send = gr.Button("Send")

    send.click(
        fn=chat_fn,
        inputs=[message, image, chatbot],
        outputs=[message, chatbot]
    )

    message.submit(
        fn=chat_fn,
        inputs=[message, image, chatbot],
        outputs=[message, chatbot]
    )

    with gr.Accordion("🧠 Agent Reasoning Trace", open=False):
        trace = gr.Textbox(
            value="Reasoning trace will appear here.",
            lines=10,
            interactive=False
        )

demo.launch()