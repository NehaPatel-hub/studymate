# ===============================
# StudyMate — AI Academic Assistant (Memory-Efficient)
# ===============================

import streamlit as st
import io
import re
import requests
from typing import List, Tuple
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# Configuration
# ----------------------------
HF_PRIMARY = "https://router.huggingface.co/hf-inference/models"
HF_FALLBACK = "https://api.huggingface.co/models"
DEFAULT_MODEL = "ibm-granite/granite-3.2-2b-instruct"

# Helsinki translation model
HELSINKI_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"  # English ↔ Spanish/French/etc.

# Chunk settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = 500  # Maximum chunks to prevent memory issues

# ----------------------------
# Streamlit UI setup & styles
# ----------------------------
st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")
st.markdown("""
    <style>
      .stApp { max-width: 1100px; margin: 0 auto; }
      .chat-bubble { padding: 14px; border-radius: 12px; margin-bottom: 12px;}
      .user { background: #e3f2fd; border-left: 6px solid #1976d2;}
      .assistant { background: #f1f8e9; border-left: 6px solid #43a047;}
      .sidebar .stButton > button { width:100%; }
      .small {font-size:0.9rem; color: #666;}
      .success-box { padding:10px; border-left:4px solid #4caf50; background:#e8f5e9; border-radius:6px; }
      .warning-box { padding:10px; border-left:4px solid #ff9800; background:#fff8e1; border-radius:6px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Session state initialization
# ----------------------------
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts: List[str] = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names: List[str] = []
if "chunks" not in st.session_state:
    st.session_state.chunks: List[Tuple[str, str]] = []  # (chunk_text, source_name)
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []  # [{'role':..., 'content':...}]

# ----------------------------
# Utility functions
# ----------------------------
def safe_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def extract_text_from_pdf_file(uploaded_file) -> str:
    try:
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        pages = []
        for p in pdf_reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        st.error(f"Failed to read PDF '{getattr(uploaded_file, 'name', '')}': {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = safe_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0: start = 0
    return chunks

def score_chunks_by_overlap(question: str, chunks: List[Tuple[str, str]], top_k: int = 4):
    q_tokens = set(re.findall(r"\w+", question.lower()))
    scored = []
    for chunk, src in chunks:
        c_tokens = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(q_tokens & c_tokens)
        scored.append((chunk, src, overlap))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

def build_prompt_with_context(question: str, top_chunks: List[Tuple[str, str, int]]) -> str:
    context_parts = []
    for chunk, src, score in top_chunks:
        context_parts.append(f"Source: {src}\n{chunk}")
    context = "\n\n---\n\n".join(context_parts) if context_parts else ""
    system_msg = ("You are an expert academic assistant. Answer concisely using ONLY the context below. "
                  "If the answer isn't present in the context, say you don't know and provide a short helpful suggestion.")
    prompt = f"{system_msg}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return prompt[:30000]

def call_hf_inference(model: str, prompt: str, token: str,
                      max_new_tokens: int = 512, temperature: float = 0.2):
    if not token:
        return False, "Missing HuggingFace token."
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt,
               "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature, "return_full_text": False}}
    # Try primary endpoint
    try:
        url_primary = f"{HF_PRIMARY}/{model}"
        resp = requests.post(url_primary, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return True, data[0]["generated_text"]
            return True, str(data)
        if resp.status_code == 503: return False, "Model loading (503). Try again."
        if resp.status_code in (401,403): return False, "Unauthorized. Check your token."
    except Exception as e:
        print("Primary failed:", e)
    # Try fallback
    try:
        url_fallback = f"{HF_FALLBACK}/{model}"
        resp = requests.post(url_fallback, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return True, data[0]["generated_text"]
            return True, str(data)
        return False, f"Fallback error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, f"Both endpoints failed: {e}"

# ----------------------------
# Translation pipeline (Helsinki)
# ----------------------------
@st.cache_resource
def get_translation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(HELSINKI_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(HELSINKI_MODEL)
    return pipeline("translation", model=model, tokenizer=tokenizer)

translator_pipeline = get_translation_pipeline()

def translate_text_helsinki(text: str) -> str:
    if not text.strip(): return ""
    result = translator_pipeline(text)
    return result[0]["translation_text"]

# ----------------------------
# Sidebar - config & upload
# ----------------------------
with st.sidebar:
    st.title("StudyMate — Configuration")
    st.text_input("HuggingFace Token", value=st.session_state.hf_token, type="password", key="token_input", help="Required for Granite model.")
    st.text_input("Granite Model Repo", value=st.session_state.model_name, key="model_input")
    st.markdown("---")
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Process PDFs"):
        st.session_state.pdf_texts = []
        st.session_state.pdf_names = []
        st.session_state.chunks = []
        progress = st.progress(0)
        for i, f in enumerate(uploaded_files):
            txt = extract_text_from_pdf_file(f)
            if txt:
                st.session_state.pdf_texts.append(txt)
                st.session_state.pdf_names.append(getattr(f,"name",f"file_{i}"))
                chunked = chunk_text(txt)
                for c in chunked:
                    st.session_state.chunks.append((c,getattr(f,"name",f"file_{i}")))
                if len(st.session_state.chunks) >= MAX_CHUNKS:
                    st.session_state.chunks = st.session_state.chunks[:MAX_CHUNKS]
                    break
            txt = None  # free memory
            progress.progress(int((i+1)/len(uploaded_files)*100))
        st.success(f"Processed {len(st.session_state.pdf_texts)} PDF(s) -> {len(st.session_state.chunks)} chunks.")

    if st.button("Clear all data"):
        st.session_state.chat_history = []
        st.session_state.pdf_texts = []
        st.session_state.pdf_names = []
        st.session_state.chunks = []
        st.experimental_rerun()

# ----------------------------
# Main Area
# ----------------------------
st.title("📚 StudyMate — AI Academic Assistant")
tabs = st.tabs(["💬 Chat", "🌐 Translate", "ℹ️ About & Tips"])

# ---- Chat Tab ----
with tabs[0]:
    st.header("Chat with Study Materials")
    for m in st.session_state.chat_history:
        cls = "user" if m["role"]=="user" else "assistant"
        st.markdown(f"<div class='chat-bubble {cls}'><strong>{m['role'].title()}:</strong><br>{m['content']}</div>", unsafe_allow_html=True)

    q_col, btn_col = st.columns([5,1])
    with q_col:
        user_question = st.text_input("Ask question about PDFs or general topic:", key="chat_input")
    with btn_col:
        send = st.button("Send")

    if send and user_question:
        st.session_state.chat_history.append({"role":"user","content":user_question})
        with st.spinner("Generating answer..."):
            if st.session_state.chunks:
                ranked = score_chunks_by_overlap(user_question, st.session_state.chunks, top_k=4)
                prompt = build_prompt_with_context(user_question, ranked)
            else:
                prompt = f"Question: {user_question}\n\nAnswer concisely:"
            ok, resp = call_hf_inference(st.session_state.model_name, prompt, st.session_state.hf_token)
            st.session_state.chat_history.append({"role":"assistant","content":resp if ok else f"[ERROR] {resp}"})
            st.experimental_rerun()

# ---- Translate Tab ----
with tabs[1]:
    st.header("Translate Text (Helsinki-NLP)")
    text_to_translate = st.text_area("Enter text", height=200)
    if st.button("Translate Text"):
        if text_to_translate.strip():
            translated_text = translate_text_helsinki(text_to_translate)
            st.text_area("Translated Text", value=translated_text, height=200)
        else:
            st.warning("Enter text to translate.")

# ---- About & Tips ----
with tabs[2]:
    st.header("About & Quick Tips")
    st.markdown("""
    - Chatbot and PDF Q&A use **IBM Granite 3.2 2B Instruct** (requires HuggingFace token).
    - Translation uses **Helsinki-NLP/opus-mt-en-ROMANCE** model for fast & accurate results.
    - Upload PDFs to ask context-aware questions.
    - Clear data or rerun if model fails to respond.
    - For scanned PDFs, text extraction may fail (use OCR first).
    """)
    st.markdown("<div class='small'>Made with ❤️ by StudyMate | Streamlit & HuggingFace</div>", unsafe_allow_html=True)
