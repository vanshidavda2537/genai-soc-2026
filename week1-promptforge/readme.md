# 🔧 PromptForge — Multi-Mode AI Assistant

A Gradio web app with 4 selectable AI personas, each with a unique system prompt,
few-shot examples, and output style. Built as Week 1 project for GenAI Summer of Code 2026.

---

## 🚀 How to Run Locally

1. Clone the repo
   git clone https://github.com/vanshidavda2537/genai-soc-2026.git
   cd genai-soc-2026/week1-promptforge

2. Create and activate a virtual environment
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux

3. Install dependencies
   pip install -r requirements.txt

4. Set up your API key
   Copy .env.example to .env and add your Groq API key:
   GROQ_API_KEY=your_key_here

5. Run the app
   python app.py

Then open http://127.0.0.1:7860 in your browser.

---

## 🎭 The 4 Personas

### 1. Technical Explainer
Explains complex topics clearly using simple language and step-by-step reasoning.
![Technical Explainer](screenshots/technicalexplainerss.png)

### 2. Debate Coach
Presents both sides of any argument fairly before concluding.
![Debate Coach](screenshots/debatecoachss.png)

### 3. Code Reviewer
Reviews code and returns structured JSON with issues, suggestions, and severity.
![Code Reviewer](screenshots/codereviewerss.png)

### 4. Creative Writer
Responds with vivid, imaginative prose and descriptive language.
![Creative Writer](screenshots/creativewriterss.png)

---

## 🧠 Concepts Used

- **System Prompts** — each persona has unique instructions sent to the AI
- **Few-Shot Examples** — sample Q&A pairs injected before the user message
- **Streaming** — responses appear token by token using Groq's stream=True
- **JSON Rendering** — Code Reviewer output parsed and displayed as Markdown
- **Temperature Control** — slider to control response creativity (0.0 to 1.5)
- **Gradio Blocks** — UI built entirely in Python

---

## 📁 Project Structure

week1-promptforge/
├── app.py              
├── requirements.txt    
├── .env.example        
├── .gitignore          
└── screenshots/        

---

## ⚙️ Requirements

- Python 3.8+
- Groq API key (free at console.groq.com)


