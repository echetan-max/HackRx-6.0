# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt
# uvicorn main:app --host 127.0.0.1 --port 8000 --reload
#http://127.0.0.1:8000/docs to open UI
import os
import io
import re
import json
import time
import hashlib
import requests
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ---- Document parsing ----
from pypdf import PdfReader
from docx import Document as DocxDocument
from email import policy
from email.parser import BytesParser

# ---- Embeddings / Similarity ----
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import faiss  # faiss-cpu
except ImportError as e:
    raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu") from e


# ============ CONFIG ============
# Hackathon base path requirement
API_PREFIX = "/api/v1"

# Team token from problem statement (can override via env)
TEAM_TOKEN = os.getenv(
    "HACKRX_TEAM_TOKEN",
    "73332fdc9c30b48a918eadc5e9a8c379e902dd1126f2bfb9024c15c6daeaff29"
)

# Free/local embeddings
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# Optional free LLM via HuggingFace Inference API (leave empty to disable)
HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")
LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))  # seconds

# Index storage
INDEX_DIR = os.getenv("INDEX_DIR", "./indices")

# Retrieval params
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6            # initial retrieve
FINAL_CONTEXT_K = 4  # context passed to LLM/extractor
TOP_K_CLAUSES = 4    # number of supporting clause snippets returned

# Toggle auth easily while debugging
DISABLE_AUTH = os.getenv("DISABLE_AUTH", "0") == "1"
# ================================


# ===== FastAPI app wiring =====
app = FastAPI(
    title="HackRx RAG – Free/Local",
    version="1.0.0",
    description="Free FAISS + local embeddings + optional HF LLM with explainable retrieval.",
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
)

# CORS (nice to have during testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# Lazy-load embedding model so startup is snappy
embedding_model = None


# ====== Models ======
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# ====== Utilities ======
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def sent_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def sliding_windows(sents: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        if cur_len + len(s) + 1 <= max_chars:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
            # overlap tail
            tail, tail_len = [], 0
            for t in reversed(cur):
                if tail_len + len(t) + 1 <= overlap_chars:
                    tail.insert(0, t)
                    tail_len += len(t) + 1
                else:
                    break
            cur = tail + [s]
            cur_len = sum(len(x) + 1 for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_clauses(text: str) -> List[str]:
    """
    Try to isolate clauses/sections for explainability and precision.
    Detects numbered headings and ALL CAPS headings. Falls back to windows.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    sections = []
    buf = []
    heading_re = re.compile(r"^(\d+(\.\d+)*|\(\w+\)|[A-Z][A-Z0-9 ]{3,})[\.:)\- ]")
    for ln in lines:
        if heading_re.match(ln) and buf:
            sections.append(" ".join(buf).strip())
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        sections.append(" ".join(buf).strip())

    if len(sections) < 4:
        # fallback to windows if we didn't detect much structure
        return sliding_windows(sent_split(text), 600, 120)
    return sections


def _get_embedder():
    global embedding_model
    if embedding_model is None:
        print("[startup] Loading embedding model…")
        t0 = time.time()
        # normalize embeddings=True gives unit vectors, so dot = cosine
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"[startup] Embedding model ready in {time.time()-t0:.1f}s")
    return embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = _get_embedder()
    embs = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")


def cosine_from_unit(a: np.ndarray, b: np.ndarray) -> float:
    # when using normalized vectors, dot == cosine
    return float(np.dot(a, b))


def read_url(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def detect_type(url: str, content: bytes) -> str:
    low = url.lower()
    if low.endswith(".pdf"):  return "pdf"
    if low.endswith(".docx"): return "docx"
    if low.endswith(".eml") or b"Content-Type: message/rfc822" in content[:200]:
        return "eml"
    if b"%PDF" in content[:10]:  # magic header
        return "pdf"
    return "text"


def extract_text_from_bytes(content: bytes, kind: str) -> str:
    if kind == "pdf":
        reader = PdfReader(io.BytesIO(content))
        txt = "".join((page.extract_text() or "") for page in reader.pages)
        return txt
    if kind == "docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if kind == "eml":
        msg = BytesParser(policy=policy.default).parsebytes(content)
        parts = []
        if msg["subject"]: parts.append(f"Subject: {msg['subject']}")
        if msg["from"]:    parts.append(f"From: {msg['from']}")
        if msg["to"]:      parts.append(f"To: {msg['to']}")
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    parts.append(part.get_content())
        else:
            if msg.get_content_type() == "text/plain":
                parts.append(msg.get_content())
        return "\n".join(parts)
    # text fallback
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def build_doc_index(doc_url: str, raw_text: str) -> Tuple[faiss.Index, Dict[str, Any], str]:
    """
    Build (or load) a FAISS index for doc_url. Stores both chunk & clause vectors.
    """
    ensure_dir(INDEX_DIR)
    key = sha1(doc_url)
    index_path = os.path.join(INDEX_DIR, f"{key}.faiss")
    meta_path = index_path + ".json"

    if os.path.exists(index_path) and os.path.exists(meta_path):
        idx = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return idx, meta, index_path

    # chunking
    sents = sent_split(raw_text)
    chunks = sliding_windows(sents, CHUNK_SIZE, CHUNK_OVERLAP)
    clauses = split_clauses(raw_text)

    all_units, unit_meta = [], []
    for i, ch in enumerate(chunks):
        all_units.append(ch)
        unit_meta.append({"type": "chunk", "id": f"chunk_{i}"})
    for j, cl in enumerate(clauses):
        all_units.append(cl)
        unit_meta.append({"type": "clause", "id": f"clause_{j}"})

    embs = embed_texts(all_units)  # normalized
    dim = embs.shape[1] if embs.size else 384
    index = faiss.IndexFlatIP(dim)  # IP = cosine when normalized
    if embs.shape[0] > 0:
        index.add(embs)

    meta = {
        "doc_url": doc_url,
        "units": all_units,
        "unit_meta": unit_meta,
        "built_at": time.time(),
        "embed_model": EMBED_MODEL_NAME,
        "params": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
    }
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return index, meta, index_path


def retrieve(index: faiss.Index, meta: Dict[str, Any], query: str, k: int = TOP_K):
    q_emb = embed_texts([query])[0]
    D, I = index.search(np.array([q_emb]), k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        unit_text = meta["units"][idx]
        unit_m = meta["unit_meta"][idx]
        out.append((unit_text, unit_m, float(score)))
    return out


def clause_rerank(candidates, query: str):
    reranked = []
    q_words = set(query.lower().split())
    for text, m, score in candidates:
        bonus = 0.0
        if m.get("type") == "clause":
            bonus += 0.05  # nudge real clauses higher
        overlap = len(q_words & set(text.lower().split()))
        bonus += min(overlap * 0.002, 0.04)
        reranked.append((text, m, score + bonus))
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked


def build_context(reranked, top_k: int = FINAL_CONTEXT_K):
    chosen = reranked[:top_k]
    context = "\n\n---\n\n".join([c[0] for c in chosen])
    support = [{
        "id": c[1]["id"],
        "type": c[1]["type"],
        "score": round(float(c[2]), 4),
        "text": c[0][:1200]
    } for c in chosen]
    return context, support


def call_hf_llm(prompt: str) -> Optional[str]:
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(LLM_API_URL, headers=headers, json={"inputs": prompt}, timeout=LLM_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            if isinstance(data, str):
                return data
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                return data["choices"][0].get("text")
        # if 503/loading/etc, return None to trigger fallback
        return None
    except requests.RequestException:
        return None


def concise_extractive_answer(question: str, context: str) -> str:
    q_emb = embed_texts([question])[0]
    sents = sent_split(context)
    if not sents:
        return "I cannot find this information in the provided document."
    s_embs = embed_texts(sents)
    sims = (s_embs @ q_emb)  # normalized vectors ⇒ dot = cosine
    idxs = np.argsort(-sims)[:4]
    picked = []
    seen = set()
    for i in idxs:
        s = sents[i].strip()
        k = s.lower()
        if k not in seen:
            picked.append(s)
            seen.add(k)
    return " ".join(picked)


def reason_rationale(question: str, clauses: List[Dict[str, Any]]) -> str:
    bits = []
    for c in clauses[:3]:
        first = " ".join(sent_split(c["text"])[:2])[:300]
        bits.append(f"[{c['id']}] {first}")
    joined = " | ".join(bits)
    return f"Based on the most relevant clauses {joined}. The answer reflects those conditions."


# ====== Auth helper (Swagger lock button works) ======
def check_auth(
    req: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if DISABLE_AUTH:
        return True
    token = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials.strip()
    else:
        # allow ?token=... during testing
        token = req.query_params.get("token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return True


# ====== Routes ======
@app.get("/", include_in_schema=False)
def root():
    # redirect to /api/v1/docs so judges see the UI immediately
    return RedirectResponse(url=f"{API_PREFIX}/docs")

@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok"}

@app.post(f"{API_PREFIX}/hackrx/run")
def run_submission(request: QueryRequest, _: bool = Depends(check_auth)):
    t0 = time.time()
    # 1) Fetch & parse document
    try:
        raw = read_url(request.documents)
        kind = detect_type(request.documents, raw)
        text = extract_text_from_bytes(raw, kind)
        if not text or len(text.strip()) < 10:
            raise ValueError("Empty or unreadable document text.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read document: {e}")

    # 2) Build/load FAISS index
    try:
        idx, meta, idx_path = build_doc_index(request.documents, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    # 3) Answer each question
    answers_out = []
    answers_legacy = []

    for q in request.questions:
        q_start = time.time()
        try:
            cands = retrieve(idx, meta, q, k=TOP_K)
            cands = clause_rerank(cands, q)
            context, supports = build_context(cands, top_k=FINAL_CONTEXT_K)

            prompt = (
                "Answer the user question ONLY using the provided context. "
                "If not answerable, say you cannot find it in the document. "
                "Be concise and precise.\n\n"
                f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
            )

            llm_ans = call_hf_llm(prompt)
            if not llm_ans:
                llm_ans = concise_extractive_answer(q, context)

            rationale = reason_rationale(q, supports)

            per_q = {
                "question": q,
                "answer": llm_ans.strip(),
                "supporting_clauses": supports[:TOP_K_CLAUSES],
                "decision_rationale": rationale,
                "latency_ms": int((time.time() - q_start) * 1000)
            }
            answers_out.append(per_q)
            answers_legacy.append(per_q["answer"])
        except Exception as e:
            # Never crash the whole batch; return graceful error per question
            err_ans = "I encountered an error processing this question."
            answers_out.append({
                "question": q,
                "answer": err_ans,
                "supporting_clauses": [],
                "decision_rationale": f"Error: {e}",
                "latency_ms": int((time.time() - q_start) * 1000)
            })
            answers_legacy.append(err_ans)

    resp = {
        # shape expected by the sample scoring UI
        "answers": answers_legacy,

        # richer explainability
        "details": answers_out,

        # telemetry/meta
        "meta": {
            "doc_url": request.documents,
            "index_path": idx_path,
            "model_embed": EMBED_MODEL_NAME,
            "model_llm": LLM_MODEL if HF_TOKEN else "extractive_fallback_only",
            "total_latency_ms": int((time.time() - t0) * 1000)
        }
    }
    return JSONResponse(resp)
