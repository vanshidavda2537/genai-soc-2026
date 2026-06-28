import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool

def add_numbers(a:int ,b:int) ->int:
    """
    Add two integers and returns the sum
    """
    return a+b

llm = ChatGroq (
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

agent=create_react_agent(
    model=llm,
    tools=[add_numbers]
)

result=agent.invoke(
    {
        "messages":[
            ("user","what is 12+15")
        ]
    }
)

print(result["messages"][-1].content)


