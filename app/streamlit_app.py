import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Agent", layout="wide")
st.title("🤖 Multi-Tool AI Agent")
st.caption("Powered by LangChain + Groq — asks documents, searches the web, and does math")

# ── Sidebar ───────────────────────────────────────────────────────────────
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

# ── Health + Tools ────────────────────────────────────────────────────────
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask anything — document, web, or math..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = f"Error: {response.json()['detail']}"
            except Exception as e:
                answer = f"Connection error: {str(e)}"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()