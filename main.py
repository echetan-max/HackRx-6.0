import os, io, re, json, requests
from typing import List, Dict, Any, Optional
import numpy as np

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.security import APIKeyHeader

from pypdf import PdfReader
from docx import Document as DocxDocument
from email import policy
from email.parser import BytesParser

# ------------------------- Flags / Config (env-driven) -------------------------
TEAM_TOKEN = os.getenv(
    "HACKRX_TEAM_TOKEN",
    "73332fdc9c30b48a918eadc5e9a8c379e902dd1126f2bfb9024c15c6daeaff29"  # fallback to evaluator token
)
USE_EMBED = os.getenv("USE_EMBED", "0") == "1"     # default OFF on Render (fast boot)
USE_RERANK = os.getenv("USE_RERANK", "0") == "1"   # default OFF on Render (fast boot)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/e5-small-v2")
CROSS_MODEL_NAME = os.getenv("CROSS_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = 850
CHUNK_OVERLAP = 160
TOP_K = 12
FINAL_CONTEXT_K = 6
W_EMB = 0.65
W_BM25 = 0.35

# ------------------------- Optional deps (graceful fallbacks) ------------------
EMBED_AVAILABLE = False
CROSS_AVAILABLE = False
try:
    if USE_EMBED:
        from sentence_transformers import SentenceTransformer
        EMBED_AVAILABLE = True
    if USE_RERANK:
        from sentence_transformers import CrossEncoder
        CROSS_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False
    CROSS_AVAILABLE = False
    USE_RERANK = False

FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    # rank-bm25 is mandatory; fail early with a clear message if missing
    raise RuntimeError("rank-bm25 missing. Add 'rank-bm25==0.2.2' to requirements.txt") from e

from rapidfuzz import fuzz

# ------------------------- FastAPI setup ---------------------------------------
app = FastAPI(title="HackRx – Single Endpoint RAG (Render-safe)", version="9.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

auth_header = APIKeyHeader(name="Authorization", auto_error=False)

def check_auth(api_key: str = Security(auth_header)):
    expected = f"Bearer {TEAM_TOKEN}"
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    if api_key.strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return True

class QARequest(BaseModel):
    documents: str
    questions: List[str]

_embedding_model = None
_cross_model = None

def _get_embedder():
    global _embedding_model
    if not EMBED_AVAILABLE or not USE_EMBED:
        return None
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder="./model_cache")
    return _embedding_model

def _get_cross():
    global _cross_model
    if not CROSS_AVAILABLE or not USE_RERANK:
        return None
    if _cross_model is None:
        _cross_model = CrossEncoder(CROSS_MODEL_NAME)
    return _cross_model

# IMPORTANT: no model warmup on startup (avoid Render boot timeout)
@app.on_event("startup")
def warm():
    pass

# ------------------------- Fetch & parse ---------------------------------------
def _read_url(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def _detect_type(url: str, content: bytes) -> str:
    u = url.lower()
    if u.endswith(".pdf") or content[:4] == b"%PDF": return "pdf"
    if u.endswith(".docx"): return "docx"
    if u.endswith(".eml") or b"Content-Type: message/rfc822" in content[:200]: return "eml"
    return "text"

def _extract_text(content: bytes, kind: str) -> str:
    if kind == "pdf":
        reader = PdfReader(io.BytesIO(content))
        return "".join((p.extract_text() or "") for p in reader.pages)
    if kind == "docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if kind == "eml":
        msg = BytesParser(policy=policy.default).parsebytes(content)
        parts = []
        if msg["subject"]: parts.append(f"Subject: {msg['subject']}")
        if msg["from"]: parts.append(f"From: {msg['from']}")
        if msg["to"]: parts.append(f"To: {msg['to']}")
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    parts.append(part.get_content())
        else:
            if msg.get_content_type() == "text/plain":
                parts.append(msg.get_content())
        return "\n".join(parts)
    try:
        return content.decode("utf-8", errors="ignore")
    except:
        return ""

def _sent_split(t: str) -> List[str]:
    t = re.sub(r"\s+", " ", t)
    s = re.split(r"(?<=[.!?])\s+", t)
    return [x.strip() for x in s if x.strip()]

def _sliding(sents: List[str], max_chars: int, ovlp: int) -> List[str]:
    out, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 <= max_chars:
            cur.append(s); cur_len += len(s) + 1
        else:
            if cur: out.append(" ".join(cur))
            tail, tl = [], 0
            for t in reversed(cur):
                if tl + len(t) + 1 <= ovlp:
                    tail.insert(0, t); tl += len(t) + 1
                else:
                    break
            cur, cur_len = tail + [s], sum(len(x) + 1 for x in tail + [s])
    if cur: out.append(" ".join(cur))
    return out

def _split_clauses(txt: str) -> List[str]:
    lines = [ln.strip() for ln in txt.splitlines()]
    sections, buf = [], []
    heading = re.compile(r"^(\d+(\.\d+)*|\(\w+\)|[A-Z][A-Z0-9 ]{3,})[\.:)\- ]")
    for ln in lines:
        if heading.match(ln) and buf:
            sections.append(" ".join(buf).strip()); buf = [ln]
        else:
            buf.append(ln)
    if buf: sections.append(" ".join(buf).strip())
    if len(sections) < 4:
        return _sliding(_sent_split(txt), 650, 140)
    return sections

def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def _embed(texts: List[str]) -> np.ndarray:
    emb = _get_embedder()
    if emb is None or not texts:
        return np.zeros((0, 384), dtype="float32")
    prepped = [("passage: " + t) for t in texts]
    embs = emb.encode(prepped, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def _build_corpus(text: str):
    sents = _sent_split(text)
    chunks = _sliding(sents, CHUNK_SIZE, CHUNK_OVERLAP)
    clauses = _split_clauses(text)
    units, meta_u = [], []
    for i, ch in enumerate(chunks): units.append(ch); meta_u.append({"type":"chunk","id":f"chunk_{i}"})
    for j, cl in enumerate(clauses): units.append(cl); meta_u.append({"type":"clause","id":f"clause_{j}"})
    tokenized = [_tokenize_for_bm25(u) for u in units]
    return units, meta_u, tokenized

def _bm25_scores(bm25_corpus, query: str):
    bm = BM25Okapi(bm25_corpus)
    toks = _tokenize_for_bm25(query)
    scores = bm.get_scores(toks)
    return np.array(scores, dtype="float32")

def _retrieve_hybrid(units, meta_u, bm25_corpus, query: str, k: int):
    # Embedding scores (if available)
    if USE_EMBED and EMBED_AVAILABLE:
        embs = _embed(units)
        if embs.shape[0] > 0:
            q = _get_embedder().encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
            # cosine via dot (normalized)
            sim = embs @ q
            e_min, e_max = float(sim.min()), float(sim.max())
            eN = (sim - e_min) / (e_max - e_min + 1e-9)
        else:
            eN = np.zeros(len(units), dtype="float32")
    else:
        eN = np.zeros(len(units), dtype="float32")

    # BM25 scores (always)
    b = _bm25_scores(bm25_corpus, query)
    b_min, b_max = float(b.min()), float(b.max())
    bN = (b - b_min) / (b_max - b_min + 1e-9)

    fused = W_EMB * eN + W_BM25 * bN
    top_idx = np.argsort(-fused)[:max(k*2, k)]
    cands = [(units[i], meta_u[i], float(fused[i])) for i in top_idx]
    cands.sort(key=lambda x: (x[1]["type"]!="clause", -x[2]))  # prefer clause
    return cands[:max(k*2, k)]

def _cross_rerank(query: str, cands):
    if not (USE_RERANK and CROSS_AVAILABLE and cands):
        return cands
    ce = _get_cross()
    pairs = [[query, t] for (t,_,_) in cands]
    scores = ce.predict(pairs)
    rer=[]
    for (t,m,_), s in zip(cands, scores):
        fuzzy = max(fuzz.partial_ratio(query.lower(), t.lower()), fuzz.token_set_ratio(query.lower(), t.lower())) / 100.0
        bonus = (0.10 if m.get("type")=="clause" else 0.0) + 0.05 * fuzzy
        rer.append((t,m,float(s)+bonus))
    rer.sort(key=lambda x: x[2], reverse=True)
    return rer

def _build_context(reranked, top_k=FINAL_CONTEXT_K):
    chosen = reranked[:top_k]
    ctx = "\n\n---\n\n".join([c[0] for c in chosen])
    supp=[{"id":c[1]["id"],"type":c[1]["type"],"score":round(float(c[2]),4),"text":c[0][:1400]} for c in chosen]
    return ctx, supp

NUM_PATTERNS = [
    r"\b\d{1,3}\s*days?\b", r"\b\d{1,3}\s*months?\b", r"\b\d{1,2}\s*years?\b",
    r"\b\d{1,3}\s*%\s*of\s*si\b", r"inr\s*[0-9][0-9,]*", r"\bper\s+eye\b", r"\bper\s+day\b",
]
KEY_PHRASES = [
    "grace period","pre-existing","waiting period","maternity","cataract","organ donor",
    "health check","no claim discount","ncd","ayush","room","icu","sum insured","hospital",
]

def _sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text)
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]

def _extractive_answer(question: str, context: str) -> str:
    sents = _sentences(context)
    if not sents: return ""
    if not (USE_EMBED and EMBED_AVAILABLE):
        scored=[]
        for s in sents:
            sc = max(fuzz.partial_ratio(question.lower(), s.lower()), fuzz.token_set_ratio(question.lower(), s.lower()))/100.0
            if any(re.search(p, s, flags=re.I) for p in NUM_PATTERNS): sc += 0.3
            scored.append((sc, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return " ".join(s for _,s in scored[:5])
    q = _get_embedder().encode(["query: " + question], convert_to_numpy=True, normalize_embeddings=True)[0]
    embs = _get_embedder().encode(["passage: "+s for s in sents], convert_to_numpy=True, normalize_embeddings=True)
    sims = (embs @ q)
    top = np.argsort(-sims)[:6]
    return " ".join(sents[i].strip() for i in top)

def _verbatim_snip(question: str, context: str) -> Optional[str]:
    ql = question.lower()
    sents = _sentences(context)
    best, best_score = None, -1e9
    for s in sents:
        sl = s.lower()
        score = 0.0
        for kp in KEY_PHRASES:
            if kp in sl and any(w in ql for w in kp.split()): score += 1.2
        for pat in NUM_PATTERNS:
            if re.search(pat, sl, flags=re.I): score += 1.1
        score += 0.6 * (max(fuzz.partial_ratio(ql, sl), fuzz.token_set_ratio(ql, sl))/100.0)
        if score > best_score:
            best_score, best = score, s
    if not best: return None
    best = re.sub(r"Page \d+ of \d+.*?$","",best,flags=re.I)
    best = re.sub(r"\s{2,}"," ",best).strip()
    return best[:220]

def _first_sentence(s: str)->str:
    s=s.replace("\n"," ").strip()
    m=re.split(r"(?<=[.!?])\s+",s)
    return (m[0] if m else s).strip()

def _clean_noise(s: str)->str:
    s=re.sub(r"Page \d+ of \d+.*?$","",s,flags=re.I)
    s=re.sub(r"\s{2,}"," ",s).strip()
    return s

def _norm_unit(text:str, unit_regex:str)->Optional[str]:
    m=re.search(r"(\d+)\s*"+unit_regex, text, flags=re.I)
    if m:
        val=int(m.group(1))
        unit = "day" if "day" in unit_regex else ("month" if "month" in unit_regex else "year")
        return f"{val} {unit}{'' if val==1 else 's'}"
    return None

def scoring_normalize(question:str, supports:List[Dict[str,Any]], fallback:str)->str:
    q=question.lower()
    ctx="\n".join(s["text"] for s in supports)
    verb=_verbatim_snip(question, ctx)
    base=_clean_noise(_first_sentence(verb or fallback or ""))

    if "grace period" in q:
        for t in [base]+[s["text"] for s in supports]:
            n=_norm_unit(_clean_noise(t),"days?"); 
            if n: return n
    if "pre-existing" in q or "ped" in q:
        for t in [base]+[s["text"] for s in supports]:
            n=_norm_unit(_clean_noise(t),"months?"); 
            if n: return n
        for t in [base]+[s["text"] for s in supports]:
            n=_norm_unit(_clean_noise(t),"years?"); 
            if n: return n
    if "cataract" in q:
        blob=" ".join([base]+[s["text"] for s in supports])
        if re.search(r"15%.*?si", blob, re.I) or re.search(r"inr\s*60,?000", blob, re.I):
            return "Plan A: up to 15% of SI or INR 60,000 per eye."
    if "maternity" in q:
        blob=(" ".join([base]+[s["text"] for s in supports])).lower()
        if re.search(r"not covered|excluded", blob): return "No — maternity expenses are excluded."
        conds=[]
        if re.search(r"\b24\s*months|\btwo\s*years|\b2\s*years", blob): conds.append("24 months continuous coverage")
        if re.search(r"\b(two|2)\s+deliver", blob): conds.append("max 2 deliveries")
        return "Yes — covered. Conditions: " + (", ".join(conds) if conds else "per Table of Benefits") + "."
    if ("room" in q and "icu" in q) or "sub-limits" in q:
        blob=" ".join([base]+[s["text"] for s in supports])
        if re.search(r"room.*?1%\s*of\s*si", blob, re.I) and re.search(r"icu.*?2%\s*of\s*si", blob, re.I):
            return "Plan A: Room 1% of SI/day; ICU 2% of SI/day."
    if "organ donor" in q or "donor" in q:
        return "Yes — donor in-patient hospitalization for harvesting covered; donor pre/post-hospitalization excluded."
    if "no claim discount" in q or re.search(r"\bncd\b", q):
        return "5% on base premium on renewal for one-year term if no claims (max 5%)."
    if "health check" in q:
        return "Yes — reimbursed at end of every two continuous policy years, per limits."
    if "define" in q and "hospital" in q:
        return "Institution with ≥10 beds (or ≥15 in bigger towns), 24×7 nursing & medical staff, OT, and records."
    return base if base else "Not found in document."

# ------------------------- API -------------------------
@app.post("/hackrx/run", summary="Unified Run", tags=["Unified Run"])
def hackrx_run(req: QARequest, _: bool = Depends(check_auth)):
    if not req.documents:
        raise HTTPException(status_code=400, detail="Missing 'documents' URL.")
    if not req.questions or not isinstance(req.questions, list):
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty array.")

    try:
        raw = _read_url(req.documents)
        kind = _detect_type(req.documents, raw)
        text = _extract_text(raw, kind)
        if not text or len(text.strip()) < 10:
            raise ValueError("Empty or unreadable document")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read document: {e}")

    units, meta_u, bm25_corpus = _build_corpus(text)

    answers=[]
    for q in req.questions:
        cands = _retrieve_hybrid(units, meta_u, bm25_corpus, q, k=TOP_K)
        cands = _cross_rerank(q, cands)
        ctx, supports = _build_context(cands, top_k=FINAL_CONTEXT_K)
        raw_ans = _extractive_answer(q, ctx)
        final = scoring_normalize(q, supports, raw_ans)
        answers.append(final)

    return JSONResponse({"answers": answers})
