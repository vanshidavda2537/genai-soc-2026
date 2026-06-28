import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt=input("System: ")
user_prompt=input("User: ")

response=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": system_prompt

        },
        {
            "role": "user",
            "content":user_prompt
        }
    ]
)

print("\nAssistant: ")
print(response.choices[0].message.content)