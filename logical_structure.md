# Logical Structure — Multi-Tool AI Agent

## System Overview

The system consists of four layers: Frontend UI, API Layer, Agent Core, and Tool Layer. Each layer communicates sequentially to process a user query and return a grounded answer.

```
┌─────────────────────────────────────────────────────┐
│                  FRONTEND (Streamlit)                │
│  - PDF upload widget                                 │
│  - Chat input / message history                      │
│  - Tool status sidebar                               │
└────────────────────┬────────────────────────────────┘
                     │ HTTP POST /ask or /upload
                     ▼
┌─────────────────────────────────────────────────────┐
│                  API LAYER (FastAPI)                 │
│  - POST /upload  → index PDF into FAISS              │
│  - POST /ask     → forward question to agent         │
│  - GET  /health  → return tool + doc status          │
│  - DELETE /document → clear FAISS index              │
└────────────────────┬────────────────────────────────┘
                     │ calls run_agent(question)
                     ▼
┌─────────────────────────────────────────────────────┐
│              AGENT CORE (LangChain ReAct)            │
│                                                      │
│  1. Receive question + chat history                  │
│  2. Build prompt: system + tools + memory + question │
│  3. LLM generates: Thought → Action → Action Input   │
│  4. Parse action → execute tool                      │
│  5. Feed Observation back to LLM                     │
│  6. Repeat until LLM writes "Final Answer"           │
│  7. Store exchange in ConversationBufferMemory        │
└────────────────────┬────────────────────────────────┘
                     │ selects and calls one of 3 tools
                     ▼
┌─────────────────────────────────────────────────────┐
│                   TOOL LAYER                         │
│                                                      │
│  Tool 1: document_search                             │
│  - Embeds query with Sentence-BERT                   │
│  - Searches FAISS index (cosine similarity)          │
│  - Returns top-3 matching chunks                     │
│                                                      │
│  Tool 2: web_search                                  │
│  - Calls DuckDuckGoSearchRun                         │
│  - Returns live internet results                     │
│                                                      │
│  Tool 3: calculator                                  │
│  - Evaluates math expression safely                  │
│  - Returns numeric result                            │
└─────────────────────────────────────────────────────┘
```

## Data Flow

### Flow 1: PDF Upload and Indexing

```
User uploads PDF via Streamlit
        │
        ▼
POST /upload (multipart form)
        │
        ▼
FastAPI saves file to data/
        │
        ▼
index_document(pdf_path) called
        │
        ▼
PyMuPDF extracts full text
        │
        ▼
Text split into 500-word chunks (sliding window)
        │
        ▼
Sentence-BERT encodes each chunk → 384-dim vectors
        │
        ▼
FAISS IndexFlatIP stores normalized vectors
        │
        ▼
doc_store = {"chunks": [...], "index": faiss_index}
        │
        ▼
API returns: {filename, chunks_created}
        │
        ▼
Streamlit displays: "Indexed N chunks ✅"
```

### Flow 2: User Question → Agent Answer

```
User types question in Streamlit chat input
        │
        ▼
POST /ask {question: "..."}
        │
        ▼
FastAPI calls run_agent(agent_executor, question)
        │
        ▼
LangChain builds ReAct prompt:
  - System: "You are an agent with these tools..."
  - Tool descriptions (name + docstring)
  - Chat history from ConversationBufferMemory
  - Current question
        │
        ▼
Groq LLM (Llama 3.3 70B) generates:
  Thought: "I need to calculate this"
  Action: calculator
  Action Input: "0.25 * 1200"
        │
        ▼
LangChain parses Action → calls calculator("0.25 * 1200")
        │
        ▼
Tool returns: "0.25 * 1200 = 300.0"
        │
        ▼
LangChain appends: Observation: "0.25 * 1200 = 300.0"
        │
        ▼
LLM generates:
  Thought: "I have the answer"
  Final Answer: "25% of 1200 is 300.0"
        │
        ▼
AgentExecutor returns {"output": "25% of 1200 is 300.0"}
        │
        ▼
FastAPI returns: {question, answer}
        │
        ▼
Streamlit displays answer in chat bubble
        │
        ▼
Exchange stored in ConversationBufferMemory
```

## Tool Selection Logic

The LLM selects tools based on their docstring descriptions. No hardcoded routing — the LLM reads the description and decides:

| Tool | Docstring trigger phrase |
|---|---|
| document_search | "Questions about something that might be in an uploaded PDF" |
| web_search | "Current events, live information, or general knowledge" |
| calculator | "Any math operations" |

## Memory Architecture

`ConversationBufferMemory` stores all exchanges as a list of HumanMessage / AIMessage objects. On each new question, the full history is injected into the prompt so the LLM has context for follow-up questions.

## State Management

- **Document state** — stored in `doc_store` dict (in-memory, reset on server restart)
- **Conversation state** — stored in `ConversationBufferMemory` (in-memory, per session)
- **No persistent database** — all state is in-memory for simplicity
