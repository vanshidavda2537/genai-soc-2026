import os
import json
import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

print("API Key loaded: ", api_key is not None)

PERSONAS = {
    "Technical Explainer": {
        "system_prompt": (
            "You are a technical explainer. "
            "Explain concepts clearly, step-by-step, using simple language."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "What is a variable?"
            },
            {
                "role": "assistant",
                "content": "A variable is a container used to store data."
            }
        ],
        "output_format": "text"
    },

    "Debate Coach": {
        "system_prompt": (
            "You are a debate coach. "
            "Present both sides of every argument fairly before concluding."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Should students use AI?"
            },
            {
                "role": "assistant",
                "content": (
                    "Pros: Faster learning and productivity. "
                    "Cons: Risk of over-dependence and reduced problem-solving practice."
                )
            }
        ],
        "output_format": "text"
    },

    "Code Reviewer": {
        "system_prompt": (
            "You are a code reviewer. "
            "Always respond ONLY in valid JSON format with exactly these keys: "
            "issues, suggestions, severity. "
            "Do not include any extra text outside the JSON."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Review this code"
            },
            {
                "role": "assistant",
                "content": '{"issues": ["No input validation"], "suggestions": ["Validate user input before processing"], "severity": "medium"}'
            }
        ],
        "output_format": "json"
    },

    "Creative Writer": {
        "system_prompt": (
            "You are a creative writer. "
            "Write vividly with imagination and descriptive language."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Describe a sunset"
            },
            {
                "role": "assistant",
                "content": (
                    "The sky melted into shades of gold and crimson as "
                    "the sun slowly disappeared beyond the horizon."
                )
            }
        ],
        "output_format": "text"
    },
}


def build_message(mode, user_message):
    persona = PERSONAS[mode]
    messages = [
        {
            "role": "system",
            "content": persona["system_prompt"]
        }
    ]
    # add few-shot examples (already in correct role/content format)
    messages += list(persona["few_shot_examples"])

    # add the real user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    return messages


def render_code_review(raw_text):
    """Parse JSON from Code Reviewer and format it nicely."""
    try:
        data = json.loads(raw_text.strip())
        output = "### 🔍 Code Review Results\n\n"
        output += f"**Severity:** `{data.get('severity', 'unknown').upper()}`\n\n"
        output += "**Issues Found:**\n"
        issues = data.get("issues", [])
        if issues:
            for issue in issues:
                output += f"- ❌ {issue}\n"
        else:
            output += "- ✅ No issues found\n"
        output += "\n**Suggestions:**\n"
        suggestions = data.get("suggestions", [])
        if suggestions:
            for s in suggestions:
                output += f"- 💡 {s}\n"
        else:
            output += "- ✅ No suggestions\n"
        return output
    except json.JSONDecodeError:
        return f"⚠️ Could not parse JSON. Raw output:\n\n{raw_text}"


def get_response(mode, user_message, temperature):
    """Stream response from Groq API token by token."""
    messages = build_message(mode, user_message)
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        stream=True
    )
    accumulated = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            accumulated += content
            yield accumulated


def show_prompt(mode):
    return PERSONAS[mode]["system_prompt"]


def respond(user_message, chat_history, mode, temperature):
    """Handle streaming response and update chat history live."""
    if not user_message.strip():
        yield chat_history
        return

    chat_history = chat_history + [{"role": "user", "content": user_message}]
    chat_history = chat_history + [{"role": "assistant", "content": ""}]

    for partial in get_response(mode, user_message, temperature):
        if mode == "Code Reviewer":
            chat_history[-1]["content"] = render_code_review(partial)
        else:
            chat_history[-1]["content"] = partial
        yield chat_history


with gr.Blocks(title="PromptForge") as demo:

    gr.Markdown("# 🔧 PromptForge — Multi-Mode AI Assistant")

    mode = gr.Dropdown(
        choices=list(PERSONAS.keys()),
        value="Technical Explainer",
        label="Choose Persona"
    )

    temperature = gr.Slider(
        minimum=0.0,
        maximum=1.5,
        value=0.7,
        step=0.1,
        label="Temperature"
    )

    with gr.Accordion("Active System Prompt", open=False):
        prompt_box = gr.Textbox(
            value=PERSONAS["Technical Explainer"]["system_prompt"],
            interactive=False
        )

    chatbot = gr.Chatbot(
        label="Chat",
        height=450,
        render_markdown=True
    )

    with gr.Row():
        message = gr.Textbox(
            label="",
            placeholder="Type your question and press Enter...",
            scale=4,
            container=False
        )
        send_btn = gr.Button("Send ➤", scale=1, variant="primary")

    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    mode.change(fn=show_prompt, inputs=mode, outputs=prompt_box)

    send_btn.click(
        fn=respond,
        inputs=[message, chatbot, mode, temperature],
        outputs=chatbot
    ).then(fn=lambda: "", outputs=message)

    message.submit(
        fn=respond,
        inputs=[message, chatbot, mode, temperature],
        outputs=chatbot
    ).then(fn=lambda: "", outputs=message)

    clear_btn.click(fn=lambda: [], outputs=chatbot)

demo.launch()
