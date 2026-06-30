from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from tools_rag import search_documents
from tools_vision import describe_image

llm = ChatGroq(
    model="openai/gpt-oss-120b",  # llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17
    temperature=0
)

duckduckgo = DuckDuckGoSearchRun()

_wiki_api = WikipediaAPIWrapper()
_wiki_query_run = WikipediaQueryRun(api_wrapper=_wiki_api)


@tool
def wikipedia(query: str) -> str:
    """Searches Wikipedia for encyclopedic/background information. Falls back
    to a DuckDuckGo search scoped to wikipedia.org if Wikipedia's own API is
    unreachable or blocked (a known issue on some ISPs)."""
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
        return f"Wikipedia lookup failed and the fallback also failed: {e}"


tools = [
    duckduckgo,
    wikipedia,
    search_documents,
    describe_image
]

state_modifier = """
You are HybridSight, an intelligent AI assistant with access to multiple tools.

You have access to EXACTLY these four tools and no others:
- duckduckgo_search
- wikipedia
- search_documents
- describe_image

Never call any tool other than these four. If none of these tools fit, answer directly without calling a tool.

Tool routing rules:

Always choose the most appropriate tool and respond in as few steps as possible.

Tool routing rules:

1. search_documents
   - Use this ONLY when the user asks questions about uploaded PDFs.

2. describe_image
   - Use this when the user asks about an uploaded image.
   - The image path will be provided in the message as [IMAGE_PATH: /path/to/image].
   - Extract the path and pass it to describe_image tool.

3. DuckDuckGo
   - Use for current events, recent news, live information.
   - Call it ONCE, then summarize. Do not loop.

4. Wikipedia
   - Use for encyclopedic knowledge.
   - Call it ONCE, then summarize. Do not loop.

5. If the question can be answered without a tool, answer directly.

Be accurate. Use each tool at most once per question.
After getting a tool result, always give a final answer immediately.
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=state_modifier
)

def chat(user_message):
    """
    Sends a user message to the HybridSight agent
    and returns the final response.
    """
    state = {
        "messages": [
            ("user", user_message)
        ]
    }

    response = agent.invoke(
        state,
        config={"recursion_limit": 25}
    )
    return response["messages"][-1].content