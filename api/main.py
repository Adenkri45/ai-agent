import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.tools import index_document, doc_store
from src.agent import create_agent, run_agent

# ── Setup ─────────────────────────────────────────────────────────────────
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Multi-Tool AI Agent", version="1.0")

# Create agent once at startup
agent_executor = create_agent()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": doc_store["index"] is not None,
        "tools": ["document_search", "web_search", "calculator"]
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = run_agent(agent_executor, req.question)

    return AnswerResponse(
        question=req.question,
        answer=answer
    )


@app.delete("/document")
def clear_document():
    doc_store["chunks"] = []
    doc_store["index"] = None
    return {"message": "Document cleared"}