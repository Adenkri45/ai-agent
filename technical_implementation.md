# Technical Implementation — Multi-Tool AI Agent

## Prerequisites

- Python 3.12
- pip
- A Groq API key (free at groq.com)

## Project Structure

```
ai-agent/
├── src/
│   ├── __init__.py
│   ├── tools.py           # Tool definitions + FAISS document store
│   └── agent.py           # LangChain ReAct agent + memory
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI REST endpoints
├── app/
│   ├── __init__.py
│   └── streamlit_app.py   # Streamlit chat UI
├── data/                  # PDF uploads stored here
├── .gitignore
├── requirements.txt
└── README.md
```

## Step 1: Environment Setup

```bash
mkdir ai-agent
cd ai-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install langchain==0.2.16 \
            langchain-groq==0.1.9 \
            langchain-community==0.2.16 \
            langchainhub==0.1.21 \
            groq \
            sentence-transformers \
            faiss-cpu \
            pymupdf \
            duckduckgo-search \
            streamlit \
            fastapi \
            uvicorn \
            python-multipart \
            requests

export GROQ_API_KEY="your-groq-api-key-here"

mkdir src api app data
touch src/__init__.py api/__init__.py app/__init__.py
```

## Step 2: Tool Layer — `src/tools.py`

This file defines the 3 tools and the in-memory FAISS document store.

```python
import os
import fitz
import faiss
import numpy as np
from langchain.tools import tool
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun

# Embedding model — loaded once, reused
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

# In-memory document store
doc_store = {
    "chunks": [],
    "index": None
}

def index_document(pdf_path: str):
    """Load PDF, chunk text, embed chunks, store in FAISS."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Chunk into 500-word windows
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), 450):
        chunk = " ".join(words[i:i+500])
        chunks.append(chunk)

    # Embed all chunks
    embedder = get_embedder()
    embeddings = embedder.encode(chunks).astype('float32')
    faiss.normalize_L2(embeddings)  # normalize for cosine similarity

    # Build FAISS index (IndexFlatIP = inner product = cosine on normalized vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # 384 dimensions
    index.add(embeddings)

    doc_store["chunks"] = chunks
    doc_store["index"] = index

@tool
def document_search(query: str) -> str:
    """Search the uploaded document for information related to the query.
    Use this when the user asks about something that might be in an uploaded PDF document."""
    if doc_store["index"] is None:
        return "No document has been uploaded yet. Please upload a PDF first."

    embedder = get_embedder()
    query_emb = embedder.encode([query]).astype('float32')
    faiss.normalize_L2(query_emb)

    scores, indices = doc_store["index"].search(query_emb, 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append(f"[Score: {score:.3f}] {doc_store['chunks'][idx][:300]}")

    return "\n\n".join(results) if results else "No relevant content found in document."

search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the internet for current information, news, or facts not in the document.
    Use this for questions about recent events, current data, or general knowledge."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations.
    Use this for any math operations. Input should be a valid math expression like '15 * 847' or '(100 + 50) / 3'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

tools = [document_search, web_search, calculator]
```

## Step 3: Agent Core — `src/agent.py`

This file creates the LangChain ReAct agent with memory.

```python
import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub
from src.tools import tools

def get_llm():
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

def create_agent():
    llm = get_llm()

    # Pull ReAct chat prompt from LangChain Hub
    # This prompt instructs the LLM to use Thought/Action/Action Input/Observation format
    prompt = hub.pull("hwchase17/react-chat")

    # Memory stores full conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # create_react_agent binds LLM + tools + prompt into a runnable agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # AgentExecutor runs the think→act→observe loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,           # prints reasoning steps to console
        max_iterations=5,       # prevents infinite loops
        handle_parsing_errors=True
    )

    return agent_executor

def run_agent(agent_executor, question: str) -> str:
    try:
        response = agent_executor.invoke({"input": question})
        return response["output"]
    except Exception as e:
        return f"Agent error: {str(e)}"
```

## Step 4: API Layer — `api/main.py`

This file exposes FastAPI endpoints for the frontend to call.

```python
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.tools import index_document, doc_store
from src.agent import create_agent, run_agent

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Multi-Tool AI Agent", version="1.0")

# Agent created once at server startup — persists across requests
agent_executor = create_agent()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.get("/health")
def health():
    """Returns server status, document load status, and available tools."""
    return {
        "status": "ok",
        "document_loaded": doc_store["index"] is not None,
        "tools": ["document_search", "web_search", "calculator"]
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, saves it to data/, calls index_document() to:
    1. Extract text with PyMuPDF
    2. Chunk into 500-word windows
    3. Embed with Sentence-BERT
    4. Store in FAISS index
    Returns chunk count on success.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    index_document(file_path)

    return {
        "message": "Document uploaded and indexed",
        "filename": file.filename,
        "chunks": len(doc_store["chunks"])
    }

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    """
    Accepts a question string, passes it to run_agent().
    Agent decides which tool to use and returns final answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = run_agent(agent_executor, req.question)

    return AnswerResponse(
        question=req.question,
        answer=answer
    )

@app.delete("/document")
def clear_document():
    """Clears the in-memory FAISS index and chunk store."""
    doc_store["chunks"] = []
    doc_store["index"] = None
    return {"message": "Document cleared"}
```

## Step 5: Frontend UI — `app/streamlit_app.py`

This file builds the chat interface using Streamlit.

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Agent", layout="wide")
st.title("🤖 Multi-Tool AI Agent")
st.caption("Powered by LangChain + Groq — asks documents, searches the web, and does math")

# ── Sidebar: PDF Upload ───────────────────────────────────────────────────
st.sidebar.header("📂 Upload Document (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Uploading and indexing..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )
    if response.status_code == 200:
        data = response.json()
        st.sidebar.success(f"✅ Indexed {data['chunks']} chunks")
        st.sidebar.info(f"📄 {data['filename']}")
    else:
        st.sidebar.error("Upload failed")

# ── Sidebar: Health + Tool Status ─────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health").json()
    st.sidebar.markdown("### 🔧 Available Tools")
    for tool in health["tools"]:
        st.sidebar.success(f"✅ {tool}")
    if health["document_loaded"]:
        st.sidebar.info("📄 Document loaded")
    else:
        st.sidebar.warning("📄 No document loaded")
except:
    st.sidebar.error("❌ API not running")

# ── Chat Interface ────────────────────────────────────────────────────────
st.subheader("💬 Chat with the Agent")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box at bottom of page
if question := st.chat_input("Ask anything — document, web, or math..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Call API and display response
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question}
                )
                answer = response.json()["answer"] if response.status_code == 200 else f"Error: {response.json()['detail']}"
            except Exception as e:
                answer = f"Connection error: {str(e)}"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat history button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
```

## Step 6: Running the Application

### Terminal 1 — Start FastAPI server:
```bash
cd ai-agent
source venv/bin/activate
export GROQ_API_KEY="your-key-here"
python -m uvicorn api.main:app --reload
# Server runs at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### Terminal 2 — Start Streamlit UI:
```bash
cd ai-agent
source venv/bin/activate
streamlit run app/streamlit_app.py
# UI runs at http://localhost:8501
```

## Step 7: Testing Each Tool

### Calculator:
```
Input: "What is 25% of 1200?"
Expected: Agent uses calculator tool → "300.0"
```

### Web Search:
```
Input: "What is the latest news about AI?"
Expected: Agent uses web_search tool → live results from DuckDuckGo
```

### Document Search:
```
1. Upload any PDF via sidebar
2. Input: "What does the document say about [topic]?"
Expected: Agent uses document_search tool → retrieves matching chunks from FAISS
```

## Key Design Decisions

| Decision | Reason |
|---|---|
| FAISS IndexFlatIP | Exact cosine similarity, no approximation, suitable for small-medium doc sizes |
| Sentence-BERT all-MiniLM-L6-v2 | Fast, lightweight, 384-dim, good semantic quality |
| LangChain ReAct | Industry standard agent pattern, handles tool selection automatically |
| Groq Llama 3.3 70B | Free API, fast inference, strong reasoning capability |
| In-memory state | Simplicity — no database setup required for single-session use |
| DuckDuckGo | Free, no API key required, privacy-respecting |

## GitHub Repository

Full source code: https://github.com/Adenkri45/ai-agent
