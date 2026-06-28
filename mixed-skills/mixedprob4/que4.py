import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

client=Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

documents=[
     "Python is a high-level programming language.",
    "LLMs are trained on vast amounts of text data.",
    "RAG combines retrieval and generation to answer questions.",
    "Embeddings convert text into vectors for semantic search.",
    "Groq provides extremely fast AI inference."
]

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_embeddings=[]

for doc in documents:
    embedding=model.encode(doc)
    document_embeddings.append(embedding)

query=input("Ask a question: ")

query_embedding=model.encode(query)

scores=[]

for embedding in document_embeddings:
    similarity=cosine_similarity(
        [query_embedding],
        [embedding]
    )[0][0]

    scores.append(similarity)

best_index=scores.index(max(scores))

retrieved_document=documents[best_index]

print("\nRetrieved Context: ")
print(retrieved_document)

system_prompt=f"""
You must answer ONLY using the context below.

Context:
{retrieved_document}

If the answer is not present in the context,
reply with: I don't know.
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
)

print("\nAnswer: ")
print(response.choices[0].message.content)


