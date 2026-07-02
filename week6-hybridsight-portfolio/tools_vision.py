# tools_vision.py
"""
Vision tool — describes / answers questions about the most recently uploaded image.

Note on the model choice: the assignment brief specifies
`llama-3.2-11b-vision-preview`, but Groq decommissioned that model a while back,
and its direct replacement (`llama-4-scout-17b-16e-instruct`) was itself marked
for deprecation on 2026-06-17. The current Groq-recommended vision model is
`qwen/qwen3.6-27b` — that's what this tool uses. If Groq deprecates that one too
by the time you read this, check https://console.groq.com/docs/vision and swap
the VISION_MODEL constant below.

qwen/qwen3.6-27b is a "thinking" model — it reasons inside <think>...</think>
tags before answering. reasoning_effort="none" tells Groq to skip that pass
entirely (Qwen3 is the one model family that actually supports turning it off);
the regex strip below is a safety net in case any of it slips through anyway.
"""
import os
import re
from groq import Groq
from langchain_core.tools import tool

VISION_MODEL = "qwen/qwen3.6-27b"

# Holds the base64 data URI of the most recently uploaded image.
# app.py sets this via set_current_image() right after the user uploads a photo.
_current_image_b64 = None


def set_current_image(data_uri: str) -> None:
    """Called by app.py whenever a new image is uploaded."""
    global _current_image_b64
    _current_image_b64 = data_uri


def clear_current_image() -> None:
    global _current_image_b64
    _current_image_b64 = None


def has_image() -> bool:
    return _current_image_b64 is not None


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()


@tool
def describe_image(question: str = "What is in this image?") -> str:
    """Describes or answers a question about the image the user most recently
    uploaded. Use this tool whenever the user asks about an uploaded picture,
    photo, screenshot, or image ("what's in this picture?", "describe the
    image", "is there text in this photo?", etc). If no image has been
    uploaded yet, this tool will say so — don't guess what might be in it."""
    global _current_image_b64

    if not _current_image_b64:
        return "No image has been uploaded yet. Ask the user to upload one first."

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": _current_image_b64},
                        },
                    ],
                }
            ],
            temperature=0.3,
            max_completion_tokens=1024,
            reasoning_effort="none",
            reasoning_format="hidden",
        )
        return _strip_thinking(completion.choices[0].message.content)
    except Exception as e:
        return (
            f"The vision tool couldn't analyze the image ({e}). "
            "Try re-uploading the image or asking again."
        )