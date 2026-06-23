import gradio as gr

from agent import (
    run_agent_with_trace,
    format_trace
)


def handle_message(message, history):

    answer, trace = run_agent_with_trace(
        message,
        "gradio-session"
    )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return (
        "",
        history,
        format_trace(trace)
    )


with gr.Blocks(title="AgentX") as demo:

    gr.Markdown("""
    # 🤖 AgentX

    Research Agent with Memory and Reasoning Trace
    """)

    chatbot = gr.Chatbot(
        label="Conversation"
    )

    trace_box = gr.Textbox(
        label="Reasoning Trace",
        lines=12,
        interactive=False
    )

    msg = gr.Textbox(
        placeholder="Ask me anything..."
    )

    send_btn = gr.Button("Send")

    send_btn.click(
        handle_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, trace_box]
    )

if __name__ == "__main__":
    demo.launch()