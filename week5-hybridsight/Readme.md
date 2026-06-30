# 🤖 HybridSight

HybridSight is a hybrid AI assistant built using **LangGraph**, **LangChain**, **Groq**, and **Gradio**. It combines Retrieval-Augmented Generation (RAG), Web Search, and Vision capabilities into a single intelligent agent.

## ✨ Features

- 📄 PDF Question Answering (RAG)
- 🌐 Live Web Search using DuckDuckGo
- 📚 Wikipedia Search
- 🖼️ Image Understanding using Groq Vision Model
- 🤖 LangGraph ReAct Agent for intelligent tool routing
- 💬 Interactive Gradio Chat Interface
- 🧠 Agent Reasoning Trace

---

## 📂 Project Structure

```
week5-hybridsight/
│
├── agent.py              # LangGraph ReAct Agent
├── app.py                # Gradio UI
├── tools_rag.py          # PDF Search Tool
├── tools_vision.py       # Vision Tool
├── requirements.txt
├── .env
├── .env.example
├── chroma_store/         # Chroma Vector Database
└── README.md
```

---

## 🚀 Technologies Used

- Python
- LangGraph
- LangChain
- Groq API
- ChromaDB
- HuggingFace Embeddings
- DuckDuckGo Search
- Wikipedia API
- Gradio

---


## 🔑 Environment Variables

Create a `.env` file and add your Groq API Key.

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

```bash
python app.py
```

The Gradio application will open in your browser.

---

## 🧩 Supported Tools

### 📄 PDF Search

Searches uploaded PDFs using ChromaDB and HuggingFace Embeddings.

---

### 🌐 DuckDuckGo

Used for current events and live web information.

---

### 📚 Wikipedia

Used for factual and encyclopedic knowledge.

---

### 🖼️ Vision Tool

Analyzes uploaded images using Groq Vision Model.

---

## 🧠 Agent Routing

The LangGraph ReAct Agent automatically decides which tool to use.

| User Query | Tool Used |
|------------|-----------|
| Question about uploaded PDF | RAG Tool |
| Current Events | DuckDuckGo |
| Historical Facts | Wikipedia |
| Uploaded Image | Vision Tool |

---

## 🧪 Test Cases

### ✅ Test Case 1

Ask a question answerable only from an uploaded PDF.

Expected Tool:

```
search_documents
```

```
screenshots/pdf_question.png
```

---

### ✅ Test Case 2

Ask about a current event.

Expected Tool:

```
DuckDuckGo
```

```
screenshots/web_search.png
```

---

### ✅ Test Case 3

Upload an image and ask:

```
What's in this image?
```

Expected Tool:

```
describe_image
```

```
screenshots/vision.png
```

---

### ✅ Test Case 4

Ask a historical or factual question.

Expected Tool:

```
Wikipedia
```

```
screenshots/wiki.png
```

---

### ✅ Test Case 5

Ask a PDF question before uploading any PDF.

Expected Output

```
Graceful "No Documents Found" message
```

```
screenshots/output.png
```

---




### DuckDuckGo Search



## 📚 Learning Outcomes

- Built a LangGraph ReAct Agent
- Implemented Retrieval-Augmented Generation (RAG)
- Integrated Web Search using DuckDuckGo
- Integrated Wikipedia Search
- Implemented Vision-based Question Answering
- Designed a multi-tool Gradio application
- Learned intelligent tool routing using LangGraph

---

## 👨‍💻 Author

**Vanshi Davda**

MSTC GenAI SOC 2026