import os
import fitz
import faiss
import numpy as np
from langchain.tools import tool
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun

# ── Embedding model (shared) ─────────────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

# ── In-memory document store ─────────────────────────────────────────────
doc_store = {
    "chunks": [],
    "index": None
}

# ── Tool 1: Document Search ───────────────────────────────────────────────
def index_document(pdf_path: str):
    """Load and index a PDF into FAISS."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Chunk
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), 450):
        chunk = " ".join(words[i:i+500])
        chunks.append(chunk)

    # Embed
    embedder = get_embedder()
    embeddings = embedder.encode(chunks).astype('float32')
    faiss.normalize_L2(embeddings)

    # Index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    doc_store["chunks"] = chunks
    doc_store["index"] = index
    print(f"Indexed {len(chunks)} chunks from {pdf_path}")


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


# ── Tool 2: Web Search ────────────────────────────────────────────────────
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the internet for current information, news, or facts not in the document.
    Use this for questions about recent events, current data, or general knowledge."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"


# ── Tool 3: Calculator ────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations.
    Use this for any math operations. Input should be a valid math expression like '15 * 847' or '(100 + 50) / 3'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# Export all tools
tools = [document_search, web_search, calculator]
