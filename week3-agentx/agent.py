import os
import datetime

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_groq import ChatGroq

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# -------------------------
# TOOL 1 : Current Date
# -------------------------

@tool
def get_current_date() -> str:
    """
    Return today's date in YYYY-MM-DD format.
    """
    return datetime.date.today().isoformat()


# -------------------------
# TOOL 2 : Web Search
# -------------------------

search = DuckDuckGoSearchRun(
    description="""
    Search the web for current events,
    recent information,
    news and facts.
    """
)

tools = [
    search,
    get_current_date
]


# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# -------------------------
# MEMORY
# -------------------------

memory = MemorySaver()


# -------------------------
# SYSTEM PROMPT
# -------------------------

system_prompt = """
You are AgentX, a research assistant.

Rules:
1. Use DuckDuckGo for factual information.
2. Use the date tool when current date is needed.
3. Mention which tool was used.
4. Be concise and helpful.
"""


# -------------------------
# AGENT
# -------------------------

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=system_prompt
)


# -------------------------
# RUN AGENT
# -------------------------

def run_agent_with_trace(user_input: str, session_id: str):

    trace_log = []
    final_answer = ""

    config = {
        "configurable": {
            "thread_id": session_id
        },
        "recursion_limit": 10
    }

    try:

        for event in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config=config,
            stream_mode="values"
        ):

            last_msg = event["messages"][-1]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:

                for tc in last_msg.tool_calls:

                    trace_log.append(
                        {
                            "tool": tc["name"],
                            "input": tc["args"]
                        }
                    )

            elif (
                hasattr(last_msg, "type")
                and last_msg.type == "ai"
                and not getattr(last_msg, "tool_calls", None)
            ):

                final_answer = last_msg.content

    except Exception as e:

        final_answer = f"Error: {str(e)}"
        trace_log.append(
            {
                "tool": "Exception",
                "input": str(e)
            }
        )

    return final_answer, trace_log


# -------------------------
# FORMAT TRACE
# -------------------------

def format_trace(trace_log):

    if not trace_log:
        return "No tools used."

    lines = []

    for i, entry in enumerate(trace_log, start=1):

        lines.append(
            f"Step {i}\n"
            f"Tool : {entry['tool']}\n"
            f"Input: {entry['input']}"
        )

    return "\n\n".join(lines)


# -------------------------
# TEST
# -------------------------

if __name__ == "__main__":

    answer, trace = run_agent_with_trace(
        "Who is Sundar Pichai?",
        "test-session"
    )

    print("\nANSWER:\n")
    print(answer)

    print("\nTRACE:\n")
    print(format_trace(trace))