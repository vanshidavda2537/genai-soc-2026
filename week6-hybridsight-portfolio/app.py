# app.py
import base64
import io
import os
import sys
import uuid

import gradio as gr
from PIL import Image

from agent import run_agent_with_trace
from tools_rag import index_documents, load_existing_store
from tools_vision import set_current_image, clear_current_image
from safe_call import safe_call

# ── Startup guard — fail loudly before the app even launches ──────────────
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY is not set. Add it to .env or HF Spaces Secrets.")
    sys.exit(1)

# Start every session fresh. chroma_store would otherwise persist between
# app restarts (locally) or testers (once deployed) -- wiping it on launch
# means each new visitor begins with an empty knowledge base, not whatever
# PDF a previous run or tester left indexed.
import shutil
if os.path.exists("./chroma_store"):
    shutil.rmtree("./chroma_store")
os.makedirs("./chroma_store", exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_pil_image(img: Image.Image) -> str:
    """Base64-encodes a PIL image as a JPEG data URI for the vision tool."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def create_session_id():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
@safe_call
@safe_call
def handle_pdf_upload(files, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload at least one PDF.")

    pdf_paths = [f.name for f in files]
    chunk_count = index_documents(
        pdf_paths,
        progress_callback=lambda frac, desc: progress(frac, desc=desc),
    )

    if chunk_count == 0:
        raise gr.Error(
            "Indexed 0 chunks -- check the PDF(s) actually contain extractable text."
        )
    return f"✅ {len(pdf_paths)} document(s) indexed -- {chunk_count} chunks stored."


@safe_call
def handle_image_upload(img):
    if img is None:
        clear_current_image()
        return "No image uploaded."
    set_current_image(encode_pil_image(img))
    return "✅ Image ready -- ask a question about it in the Image Studio tab."


@safe_call
def handle_chat(user_input, chat_history, session_id):
    if not user_input.strip():
        return chat_history, "", "Please enter a question."

    final_answer, trace_log = run_agent_with_trace(user_input, session_id)

    chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": final_answer},
    ]
    trace_text = "\n".join(trace_log) if trace_log else "No tools were called."
    return chat_history, "", trace_text


@safe_call
def handle_doc_chat(user_input, chat_history, session_id):
    """Same agent as the main chat -- the system prompt already tells it to
    prioritise search_documents, so routing a 'document QA' tab through the
    same agent keeps a single source of truth instead of a second one."""
    if not user_input.strip():
        return chat_history, ""

    final_answer, _ = run_agent_with_trace(user_input, session_id)
    chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": final_answer},
    ]
    return chat_history, ""


@safe_call
def handle_image_question(question, img):
    if img is None:
        raise gr.Error("Please upload an image first.")
    if not question.strip():
        question = "What is in this image?"

    set_current_image(encode_pil_image(img))
    final_answer, _ = run_agent_with_trace(question, "image-studio-session")
    return final_answer


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#1a5f70",
    button_primary_background_fill_hover="#154d5c",
    button_primary_text_color="#ffffff",
)

existing_chunks = 0  # always start with an empty knowledge base

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="HybridSight — GenAI Portfolio App", theme=theme) as demo:

    session_id = gr.State(value=create_session_id)

    gr.Markdown(
        """# 👁️ HybridSight
        *A hybrid GenAI agent — documents, web search, Wikipedia, and vision in one app.*"""
    )

    with gr.Tabs():

        # ── TAB 1: Main Hybrid Chat ─────────────────────────────────────
        with gr.Tab("💬 Hybrid Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=440,
                        label="Conversation",
                        placeholder="Ask anything -- documents, the web, Wikipedia, or general knowledge.",
                    )
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Ask anything...",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Column(scale=1):
                    with gr.Accordion("🔍 Reasoning Trace", open=False):
                        trace_box = gr.Textbox(
                            lines=14, interactive=False, show_label=False,
                            placeholder="Tool calls will appear here after you ask a question...",
                        )

            send_btn.click(
                handle_chat,
                inputs=[msg_box, chatbot, session_id],
                outputs=[chatbot, msg_box, trace_box],
            )
            msg_box.submit(
                handle_chat,
                inputs=[msg_box, chatbot, session_id],
                outputs=[chatbot, msg_box, trace_box],
            )

        # ── TAB 2: Document QA ──────────────────────────────────────────
        with gr.Tab("📄 Document QA"):
            with gr.Row():
                pdf_upload = gr.File(
                    file_count="multiple", file_types=[".pdf"], label="Upload PDF(s)"
                )
                index_status = gr.Textbox(
                    label="Indexing status",
                    interactive=False,
                    value="No documents indexed yet.",
                )
            index_btn = gr.Button("⟳ Index Documents", variant="primary")
            index_btn.click(handle_pdf_upload, inputs=[pdf_upload], outputs=[index_status])

            doc_chatbot = gr.Chatbot(height=380, label="Document Q&A")
            doc_input = gr.Textbox(
                placeholder="Ask about the uploaded document...", show_label=False
            )
            doc_ask_btn = gr.Button("Ask", variant="primary")

            doc_ask_btn.click(
                handle_doc_chat, inputs=[doc_input, doc_chatbot, session_id],
                outputs=[doc_chatbot, doc_input],
            )
            doc_input.submit(
                handle_doc_chat, inputs=[doc_input, doc_chatbot, session_id],
                outputs=[doc_chatbot, doc_input],
            )

        # ── TAB 3: Image Studio ─────────────────────────────────────────
        with gr.Tab("🖼️ Image Studio"):
            with gr.Row():
                img_upload = gr.Image(label="Upload image", type="pil")
                with gr.Column():
                    img_question = gr.Textbox(
                        label="Question about the image",
                        placeholder="What is in this image?",
                    )
                    img_output = gr.Textbox(label="Vision analysis", lines=8, interactive=False)
            analyse_btn = gr.Button("Analyse Image", variant="primary")
            analyse_btn.click(
                handle_image_question, inputs=[img_question, img_upload], outputs=[img_output]
            )

    gr.Markdown(
        "<center><small>HybridSight · ChromaDB RAG · DuckDuckGo · Wikipedia · "
        "Vision (Groq qwen3.6-27b) · LangGraph · Gradio</small></center>"
    )


if __name__ == "__main__":
    demo.launch(share=False)