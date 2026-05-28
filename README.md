# 🤖 Multi-Tool AI Agent

A conversational AI agent that autonomously decides which tool to use based on your question — document search, live web search, or calculator — powered by LangChain and Groq.

## 🎯 What It Does

> "Stop using one tool for everything — let the agent pick the right one"

Instead of hardcoding a fixed pipeline, this agent uses ReAct (Reasoning + Acting) to think through each question and select the appropriate tool dynamically.

---

## 🔧 Tools Available

| Tool | When Agent Uses It |
|---|---|
| 📄 Document Search | Questions about an uploaded PDF |
| 🌐 Web Search | Current events, live information |
| 🧮 Calculator | Any math operations |

---

## 💬 Example Interactions

You: "What is 25% of 1200?"
Agent: uses Calculator → "300.0"
You: "What is the latest news about AI?"
Agent: uses Web Search → fetches live results
You: "What does the document say about tax compliance?"
Agent: uses Document Search → retrieves from PDF via FAISS

---

## 🏗️ Architecture

User Question
↓
LangChain ReAct Agent
↓
Thinks: which tool do I need?
↓
┌─────────────────────────────────┐
│  Tool 1: Document Search (FAISS)│
│  Tool 2: Web Search (DuckDuckGo)│
│  Tool 3: Calculator             │
└─────────────────────────────────┘
↓
Executes tool → observes result
↓
Generates grounded answer
↓
Remembers conversation (Memory)

---

## 📁 Project Structure
├── src/
│   ├── tools.py        # 3 tool definitions
│   └── agent.py        # LangChain ReAct agent + memory
├── api/
│   └── main.py         # FastAPI endpoints
├── app/
│   └── streamlit_app.py # Chat UI
└── requirements.txt

---

## 🚀 Setup

```bash
git clone https://github.com/Adenkri45/ai-agent.git
cd ai-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
```

---

## 🧪 Running the Project

### Start FastAPI (Terminal 1)
```bash
python -m uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

### Start Streamlit (Terminal 2)
```bash
streamlit run app/streamlit_app.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask the agent a question |
| DELETE | `/document` | Clear loaded document |
| GET | `/health` | Health check + tool status |

---

## 🧠 Key Concepts

- **ReAct** — agent alternates between Reasoning and Acting until it has a final answer
- **Tool Use** — LLM selects tools based on their descriptions
- **Memory** — ConversationBufferMemory keeps track of chat history
- **FAISS** — vector similarity search for document retrieval

---

## 🛠️ Tech Stack

`Python` `LangChain` `Groq` `Llama 3.3 70B` `FAISS` `Sentence-BERT` `FastAPI` `Streamlit` `DuckDuckGo`