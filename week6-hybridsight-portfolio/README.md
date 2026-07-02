---
title: HybridSight
emoji: 👁️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
pinned: false
---

# 👁️ HybridSight — GenAI Portfolio App

*A hybrid LangGraph agent — RAG over your own documents, live web search, Wikipedia, and vision — wrapped in a polished multi-tab UI and deployed live on Hugging Face Spaces.*

**🔗 Live demo:** [huggingface.co/spaces/dhanyamankad/hybridsight](https://huggingface.co/spaces/dhanyamankad/hybridsight)

![HybridSight screenshot](screenshots/test1_empty_input.png)

---

## What This Is

This is the Week 6 "ship it" deliverable for MSTC's GenAI Track 2026 — taking Week 5's HybridSight (a 4-tool hybrid agent) and polishing it into something recruiter-ready: a 3-tab UI, real progress feedback during PDF indexing, error handling that never shows a raw traceback, and a live public URL.

The agent itself — `agent.py`, `tools_rag.py`, `tools_vision.py` — is **unchanged from Week 5**. Everything new this week lives in `app.py` (UI restructure) and a new `safe_call.py` (error handling).

## 🗂️ The Three Tabs

| Tab | What it does |
|-----|---------------|
| 💬 **Hybrid Chat** | The main agent loop — routes between documents, web search, Wikipedia, and vision automatically, with a collapsible reasoning trace panel. |
| 📄 **Document QA** | Upload PDFs, watch a real multi-stage progress bar during indexing, then ask questions grounded in the document. |
| 🖼️ **Image Studio** | Upload an image and ask a question about it (or leave it blank for a default description). |

## 🛠️ Tools the Agent Uses

| Tool | Purpose |
|------|---------|
| `search_documents` | RAG over uploaded PDFs (ChromaDB), returns chunks with source + page number |
| `describe_image` | Describes/answers questions about the uploaded image via a Groq vision model |
| `DuckDuckGoSearchRun` | Live web search for current events and recent news |
| `wikipedia_search` | Encyclopaedic/historical facts, with a DuckDuckGo-scoped fallback |

## ⚠️ Deviations From the Brief (and why)

- **Vision model:** the original Week 5 brief specified `llama-3.2-11b-vision-preview`, deprecated by Groq. This uses `qwen/qwen3.6-27b` instead, with `reasoning_effort="none"` set since it's a "thinking" model that would otherwise leak `<think>` tags into answers.
- **`safe_call` raises `gr.Error`, not a string return.** The Week 6 brief's reference implementation has `safe_call` *return* an error string on failure. That breaks the moment it wraps a handler with a multi-value output tuple (like `handle_chat`, which returns `(history, message_box, trace)`) — Gradio would crash on an output-count mismatch, the opposite of what the decorator is for. This version raises `gr.Error(message)` instead, which Gradio catches natively as a toast notification regardless of the wrapped function's return shape.
- **Fresh session on every launch.** `chroma_store/` is wiped at startup so every visitor (or every local restart) begins with an empty knowledge base rather than seeing a previous tester's uploaded PDF.
- **Granular indexing progress.** `index_documents()` in `tools_rag.py` takes an optional `progress_callback` parameter (defaults to `None`, fully backward-compatible) so the progress bar reports real stages — reading, chunking, loading the embedding model, embedding, done — instead of jumping from "started" straight to "finished."

## 🧪 Test Scenarios (all verified on the live deployment)

| # | Scenario | Expected Behaviour | Screenshot |
|---|----------|----------------------|------------|
| 1 | Click Send with an empty textbox | No crash, no action | `screenshots/test1_empty_input.png` |
| 2 | "Who was Ada Lovelace?" | Routed to `wikipedia_search`, real answer + trace | `screenshots/test2_wikipedia.png` |
| 3 | Document-specific question, no PDF uploaded | `search_documents` reports no documents indexed | `screenshots/test3_empty_kb.png` |
| 4 | Upload a PDF, click Index Documents | Progress bar visibly moves through each stage | `screenshots/test4_progress_bar.png` |
| 5 | Upload an image, leave question blank, Analyse | Default description returned | `screenshots/test5_vision.png` |

## 🚀 Local Setup

```bash
git clone <this-repo-url>
cd week6-hybridsight-portfolio
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your real key:
```
GROQ_API_KEY=your_real_key_here
```

Run it:
```bash
python app.py
```

## ☁️ Deployment

Deployed on **Hugging Face Spaces**, connected to this GitHub repo — every push to `main` triggers an automatic rebuild. `GROQ_API_KEY` is stored as a Space **Repository Secret**, never committed to git.

## 🧠 How It Works

- **Indexing** — chunk (500 chars, 100 overlap) → embed (`all-MiniLM-L6-v2`) → store (ChromaDB), same pipeline as Week 2.
- **Vision** — the uploaded image is base64-encoded client-side and held as the agent's "current image"; `describe_image` reads from that rather than the LLM passing image bytes as a tool argument.
- **Routing** — the system prompt in `agent.py` gives explicit priority rules and the ReAct agent decides which tool(s) to call per question.
- **Error handling** — every Gradio handler is wrapped in `@safe_call`, which catches `GraphRecursionError`, `groq.RateLimitError`, `groq.APIConnectionError`, `ValueError`, and anything else, converting all of them into a `gr.Error` toast instead of a raw traceback.
- **Memory** — each browser tab gets a UUID session id (`gr.State`) used as the LangGraph `thread_id`, so follow-ups have context.

## 📦 Tech Stack

[LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://github.com/langchain-ai/langchain) · [ChromaDB](https://www.trychroma.com/) · [sentence-transformers](https://www.sbert.net/) · [Groq](https://groq.com) · [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) · [Wikipedia](https://pypi.org/project/wikipedia/) · [Gradio](https://gradio.app) · [Hugging Face Spaces](https://huggingface.co/spaces)