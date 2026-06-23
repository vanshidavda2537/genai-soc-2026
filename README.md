# 🚀 GENAI-SOC 2026 Journey

Welcome to my repository for the **MSTC Summer of Code 2026 – Generative AI Track**.
This repository contains my weekly progress, projects, experiments, and learnings throughout the program.

---

## 👨‍💻 About Me

Hi, I'm **Vanshi Davda**, a second-year engineering student passionate about:

* Artificial Intelligence
* Software Development
* Problem Solving
* Building Real-World Projects

This repository documents my journey of learning and building with Generative AI.

---

# 📅 Week 0 — Environment Setup & Foundations

### Objectives Completed

* ✅ Installed Python and VS Code
* ✅ Configured Python Extension
* ✅ Created and managed Virtual Environments
* ✅ Learned Git & GitHub basics
* ✅ Set up Google Colab
* ✅ Created GitHub Repository
* ✅ Generated and tested Groq API Key
* ✅ Successfully made first API call

### Key Learnings

* Python Development Environment
* Virtual Environments
* Git Workflow
* GitHub Repositories
* API Keys & Environment Variables

---

# 🤖 Week 1 — PromptForge: Multi-Persona AI Assistant

## Project Overview

PromptForge is a multi-persona AI assistant built using **Python**, **Groq**, and **Gradio**.

The application demonstrates core prompt engineering techniques by allowing users to switch between different AI personalities, each with its own behavior, examples, and response style.

---

## ✨ Features

### 🔹 Technical Explainer

Explains technical concepts in beginner-friendly language.

### 🔹 Debate Coach

Presents balanced arguments and multiple perspectives.

### 🔹 Code Reviewer

Analyzes code and returns structured JSON feedback.

### 🔹 Creative Writer

Generates imaginative and descriptive content.

---

## 🛠 Technologies Used

* Python
* Groq API
* Llama 3.3 70B Versatile
* Gradio
* dotenv
* JSON

---

# 📚 Week 2 – DocBuddy Pro (RAG-based Multi-PDF Q&A)

## Overview

DocBuddy Pro is a Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and ask questions based on their contents.

Instead of relying solely on the LLM's training knowledge, the system retrieves relevant information from uploaded documents and uses it to generate accurate answers.

---

## Features

* 📄 Multi-PDF Upload Support
* ✂️ Document Chunking
* 🔎 Semantic Search
* 🧠 Embedding-based Retrieval
* 📚 Chroma Vector Database
* 🤖 Groq LLM Integration
* 🌐 Gradio Interface

---

## Tech Stack

* Python
* LangChain
* HuggingFace Embeddings
* ChromaDB
* Groq
* Gradio
* PyPDF
* dotenv

---

## Architecture

```text
PDF Documents
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
Chroma Vector Store
      │
      ▼
Retriever
      │
      ▼
Groq LLM
      │
      ▼
Answer
```

---

## Key Concepts Learned

* Retrieval-Augmented Generation (RAG)
* Embeddings
* Vector Databases
* Semantic Search
* Document Chunking
* LangChain Pipelines

---

# 🤖 Week 3 – AgentX (AI Agent with Tools & Memory)

## Overview

AgentX is an AI Research Assistant built using LangGraph and LangChain. Unlike a traditional chatbot, AgentX can reason, use tools, maintain memory, and display a reasoning trace showing how answers were generated.

The agent uses external tools such as web search and date retrieval to provide more accurate responses.

---

## Features

* 🔍 DuckDuckGo Search Tool
* 📅 Current Date Tool
* 🧠 Conversation Memory
* 📜 Reasoning Trace
* 🤖 ReAct Agent Architecture
* 🌐 Gradio Chat Interface

---

## Tech Stack

* Python
* LangChain
* LangGraph
* Groq
* DuckDuckGo Search
* Gradio
* dotenv

---

## Architecture

```text
User Query
     │
     ▼
AgentX
     │
     ├── DuckDuckGo Search
     ├── Current Date Tool
     ├── Memory
     │
     ▼
Reasoning
     │
     ▼
Final Answer
```

---

## Tools Used

### DuckDuckGo Search

Used for:

* Current events
* Recent news
* Real-time information
* General factual queries

### Current Date Tool

Used for:

* Date-based questions
* Time-sensitive reasoning

---

## Reasoning Trace

AgentX displays the reasoning process by showing:

* Tool selected
* Tool input
* Order of execution

Example:

```text
Step 1
Tool : duckduckgo_search
Input : latest ISRO mission

Step 2
Tool : get_current_date
Input : {}
```

---

## Key Concepts Learned

* AI Agents
* Tool Calling
* LangGraph
* Memory Systems
* ReAct Architecture
* Multi-step Reasoning
* Agent Workflows

---


## ⭐ Repository Purpose

This repository serves as a record of my progress throughout MSTC Summer of Code 2026 and showcases the projects and skills I develop during the program.
