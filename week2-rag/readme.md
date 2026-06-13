# Week 2 - Grounded PDF Q&A System (RAG)

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, HuggingFace Embeddings, Groq LLM, and Gradio.

The system allows users to:

* Upload one or more PDF documents.
* Index PDF contents into a ChromaDB vector database.
* Generate embeddings using the `all-MiniLM-L6-v2` model.
* Retrieve the most relevant chunks for a user query.
* Generate grounded answers using Groq's Llama model.
* Display retrieved context to improve transparency and reduce hallucinations.

---

## Technologies Used

* Python
* Gradio
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API
* PyPDF
* Sentence Transformers

---

## Project Workflow

1. Upload PDF documents.
2. Extract text using `PyPDFLoader`.
3. Split text into chunks of 500 characters with 100-character overlap.
4. Generate embeddings using `all-MiniLM-L6-v2`.
5. Store embeddings in ChromaDB.
6. User asks a question.
7. Retrieve top relevant chunks from ChromaDB.
8. Send retrieved context and question to Groq LLM.
9. Generate grounded response with citations.

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---


## Running the Application

```bash
python task4.py
```

The Gradio application will launch locally in the browser.

---

# Screenshot of Gradio App

![Gradio App](screenshot.png)

---

# Anti-Hallucination Testing

## Test 1: In-Scope Question

### Question

What is this project based on?

### Result

The system successfully answered using information retrieved from the uploaded PDF and provided source references.

### Observation

The response was grounded in the retrieved document context.

---

## Test 2: Out-of-Scope Question

### Question

What is the capital of France?

### Result

The system responded:

> I don't have that information in the uploaded documents.

### Observation

The model correctly avoided hallucination and followed the grounding instructions.

---

# Chunking and Embeddings 

Chunking is the process of breaking a large document into smaller pieces so that the language model can process and search the information efficiently. Instead of searching an entire PDF, the system searches through smaller chunks of text.

Embeddings are numerical vector representations of text. Similar meanings produce similar vectors. By converting both document chunks and user questions into embeddings, the system can find the most relevant chunks using similarity search.

Together, chunking and embeddings enable efficient retrieval of relevant information from large documents.

---




