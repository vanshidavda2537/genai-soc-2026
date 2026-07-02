# agent.py
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools_rag import search_documents
from tools_vision import describe_image

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

duckduckgo_tool = DuckDuckGoSearchRun(
    name="DuckDuckGoSearchRun",
    description=(
        "Use for real-time or recent information: current events, latest news, "
        "prices, recent developments, or anything that changes frequently. "
        "Prefer this over Wikipedia for anything recent."
    ),
)

# Week 3 noted that Wikipedia's own API is occasionally blocked on some Indian
# ISPs, so the raw WikipediaQueryRun call is wrapped with a DuckDuckGo-backed
# fallback scoped to wikipedia.org rather than silently failing.
_wiki_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
_wiki_query_run = WikipediaQueryRun(api_wrapper=_wiki_api)


@tool
def wikipedia_search(query: str) -> str:
    """Use for background, historical, or encyclopaedic information: biographies,
    scientific concepts, historical events, organisations, definitions, and
    foundational facts unlikely to change. Prefer this over DuckDuckGo for
    general-knowledge questions."""
    try:
        result = _wiki_query_run.run(query)
        if result and result.strip():
            return result
    except Exception:
        pass

    try:
        fallback = DuckDuckGoSearchRun().run(f"{query} site:wikipedia.org")
        return fallback or "No information found for that query."
    except Exception as e:
        return f"Wikipedia lookup failed and the fallback search also failed: {e}"


tools = [duckduckgo_tool, wikipedia_search, search_documents, describe_image]

# ═══════════════════════════════════════════════════════════════════════════
# LLM + MEMORY
# ═══════════════════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
    model_kwargs={"parallel_tool_calls": False},
)

memory = MemorySaver()

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — routing rules
# ═══════════════════════════════════════════════════════════════════════════

today = datetime.now().strftime("%A, %B %d, %Y")

SYSTEM_PROMPT = f"""You are HybridSight, a hybrid research assistant with four tools.
Today is {today}.

Routing rules — pick the right tool for each part of the question, and use more
than one tool in the same turn if the question genuinely needs it:

1. search_documents — try this FIRST whenever a PDF has been indexed and the
   question could plausibly be answered from it (document-specific facts,
   anything the user uploaded). If it reports no documents are indexed, tell
   the user plainly instead of guessing from your own knowledge.
2. describe_image — use whenever the user asks about an uploaded image, photo,
   or screenshot. If it reports no image was uploaded, say so.
3. DuckDuckGoSearchRun — use for current events, recent news, prices, or
   anything that changes frequently or happened recently.
4. wikipedia_search — use for general knowledge, historical facts, biographies,
   and other encyclopaedic information unlikely to change.

After calling a tool, write a COMPLETE answer to the user's actual question
using the real facts the tool returned. Never reply with only a meta-statement
like "This information is based on the Wikipedia search results" — that
sentence by itself is NOT an answer. State the substantive content first; you
may add which tool it came from in the same or a following sentence.

If a tool reports a limitation (no documents, no image, no results), relay
that honestly instead of fabricating an answer."""

# ═══════════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════════

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

RECURSION_LIMIT = 12


# ═══════════════════════════════════════════════════════════════════════════
# Streaming trace function (same pattern as Week 3's AgentX)
# ═══════════════════════════════════════════════════════════════════════════
def run_agent_with_trace(user_input: str, session_id: str):
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    messages = {"messages": [("user", user_input)]}
    trace_log = []
    final_answer = ""
    seen = set()

    try:
        for event in agent.stream(messages, config=config, stream_mode="updates"):
            for node_name, node_data in event.items():
                for msg in node_data.get("messages", []):
                    msg_type = type(msg).__name__

                    if msg_type == "AIMessage":
                        tool_calls = msg.additional_kwargs.get("tool_calls", [])
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args = tc["function"]["arguments"]
                            entry = f"🔧 Tool Called: {name}\n   Input: {args}"
                            if entry not in seen:
                                seen.add(entry)
                                trace_log.append(entry)

                        if msg.content and not tool_calls:
                            final_answer = msg.content

                    elif msg_type == "ToolMessage":
                        entry = f"✅ Result from {msg.name}:\n   {str(msg.content)[:200]}"
                        if entry not in seen:
                            seen.add(entry)
                            trace_log.append(entry)

    except Exception as e:
        final_answer = f"⚠️ Error: {e}"
        trace_log.append(f"❌ Exception: {e}")

    if not final_answer:
        final_answer = (
            "I wasn't able to produce a final answer for that — check the "
            "reasoning trace below for what happened."
        )

    return final_answer, trace_log