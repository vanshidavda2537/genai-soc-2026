# safe_call.py
"""
Universal exception-to-message wrapper for Gradio event handlers.

Deviation from the Week 6 brief: rather than RETURNING an error string,
this raises gr.Error(message). Gradio's own handlers normally return a
tuple matching their `outputs=[...]` list (e.g. handle_chat returns
3 values). If safe_call instead returned a single string on error, the
output count would mismatch whatever the wrapped function normally
returns and Gradio would crash anyway -- the opposite of what this
decorator exists to prevent. gr.Error sidesteps this entirely: Gradio
catches it natively, shows a clean toast notification, and leaves the
UI's existing state untouched (e.g. your chat history isn't wiped).
"""
import functools
import traceback

import gradio as gr
from langgraph.errors import GraphRecursionError
import groq


def safe_call(func):
    """Decorator: catches known failure modes (and anything else) and
    raises a Gradio-native error toast instead of letting a raw
    traceback reach the user. Apply to any Gradio handler that calls
    an LLM, a tool, or an external API."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GraphRecursionError:
            raise gr.Error(
                "The agent ran out of steps answering this. Try rephrasing "
                "your question or breaking it into smaller parts."
            )
        except groq.RateLimitError:
            raise gr.Error("Rate limit hit. Please wait a few seconds and try again.")
        except groq.APIConnectionError:
            raise gr.Error("Could not reach Groq. Check your internet connection.")
        except ValueError as e:
            raise gr.Error(f"Input error: {e}")
        except Exception as e:
            # Still log the real traceback server-side for debugging --
            # just never show it to the person using the app.
            print(f"[safe_call] Unhandled exception in {func.__name__}:")
            traceback.print_exc()
            raise gr.Error(f"Something went wrong: {e}")

    return wrapper