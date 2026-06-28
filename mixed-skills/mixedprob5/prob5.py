import os
from datetime import datetime
import gradio as gr

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

@tool
def get_current_date() ->str:
    """
    Return today's date in YYYY-MM-DD format.
    """
    return datetime.now().strftime("%Y-%m-%d")

llm=ChatGroq(
    model="llama-3.3-70b-versatile"
)

memory=MemorySaver()

agent=create_react_agent(
    model=llm,
    tools=[get_current_date],
    checkpointer=memory
)

def chat_fn(message, history):

    result = agent.invoke(
        {
            "messages": [
                (
                    "system",
                    """
                    You are a helpful assistant.

                    Whenever the user asks about
                    today's date, current date,
                    day, or calendar date,
                    use the get_current_date tool.
                    """
                ),
                ("user", message)
            ]
        },
        config={
            "configurable": {
                "thread_id": "demo-session"
            }
        }
    )

    return result["messages"][-1].content




demo = gr.ChatInterface(
    fn=chat_fn,
    title="Date Agent",
    description="LangGraph Agent with Memory + Date Tool"
)


if __name__ == "__main__":
    demo.launch()