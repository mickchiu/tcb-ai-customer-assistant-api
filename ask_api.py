# ask_api.py  v7.0.0
# -*- coding: utf-8 -*-
"""
TCB AI Customer Assistant - Ultra-Light API
Only search, never generate answers.
Let Copilot Studio AI handle the talking.
"""

import json
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
VERSION = "7.0.0"
KNOWLEDGE_FILE = Path(__file__).resolve().parent / "tcb_ai_knowledge_v6.json"
DEFAULT_TOP_K = 5
MAX_CONTENT_LEN = 600

# ============================================================
# GLOBAL STATE
# ============================================================
knowledge_data = []
tfidf_vectorizer = None
tfidf_matrix = None

# ============================================================
# TEXT UTILS
# ============================================================
def clean_text(text: str) -> str:
    """Remove HTML tags and extra whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_search_text(item: dict) -> str:
    """
    Build weighted search text for TF-IDF indexing.
    FAQ question x3, title x2, content[:600] x1
    """
    parts = []

    faq_q = item.get("faq_question", "")
    if faq_q:
        parts.extend([faq_q] * 3)

    title = item.get("title", "")
    if title:
        parts.extend([title] * 2)

    content = clean_text(item.get("content", ""))
    if content:
        parts.append(content[:MAX_CONTENT_LEN])

    return " ".join(parts)


# ============================================================
# KNOWLEDGE LOADING & INDEXING
# ============================================================
def load_knowledge():
    """Load knowledge base JSON file."""
    global knowledge_data
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)
    print(f"[INFO] Loaded {len(knowledge_data)} knowledge chunks")


def build_tfidf_index():
    """Build TF-IDF index from knowledge base."""
    global tfidf_vectorizer, tfidf_matrix

    corpus = [build_search_text(item) for item in knowledge_data]

    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        max_features=15000,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    print(f"[INFO] TF-IDF index built: {tfidf_matrix.shape}")


# ============================================================
# SEARCH
# ============================================================
def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """
    Search knowledge base using TF-IDF cosine similarity.
    Returns list of matched snippets with scores.
    """
    if tfidf_vectorizer is None or tfidf_matrix is None:
        return []

    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # Get top_k indices sorted by score descending
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        item = knowledge_data[idx]
        content = clean_text(item.get("content", ""))

        results.append({
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "content": content[:MAX_CONTENT_LEN],
            "faq_question": item.get("faq_question", ""),
            "faq_answer": item.get("faq_answer", ""),
            "source_url": item.get("source_url", ""),
            "page_type": item.get("page_type", ""),
            "score": round(score, 4),
        })

    return results


# ============================================================
# FASTAPI APP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load knowledge + build index."""
    t0 = time.time()
    load_knowledge()
    build_tfidf_index()
    print(f"[INFO] Ready in {time.time() - t0:.2f}s")
    yield
    print("[INFO] Shutting down")

app = FastAPI(
    title="TCB AI Customer Assistant API",
    description="Ultra-light: search only, no answer generation",
    version=VERSION,
    lifespan=lifespan,
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


class SnippetResponse(BaseModel):
    id: str
    title: str
    content: str
    faq_question: str
    faq_answer: str
    source_url: str
    page_type: str
    score: float


class AskResponse(BaseModel):
    matched: bool
    query: str
    count: int
    snippets: list


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    """Service info."""
    return {
        "service": "TCB AI Customer Assistant",
        "version": VERSION,
        "mode": "ultra-light (search only)",
        "knowledge_chunks": len(knowledge_data),
    }


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "version": VERSION,
        "knowledge_loaded": len(knowledge_data) > 0,
        "index_ready": tfidf_matrix is not None,
    }


@app.post("/ask")
def ask_post(req: AskRequest):
    """Main endpoint (POST): search knowledge base."""
    snippets = semantic_search(req.question, req.top_k)
    return {
        "matched": len(snippets) > 0,
        "query": req.question,
        "count": len(snippets),
        "snippets": snippets,
    }


@app.get("/ask")
def ask_get(
    q: str = Query(..., min_length=1, description="User question"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20),
):
    """Main endpoint (GET): search knowledge base."""
    snippets = semantic_search(q, top_k)
    return {
        "matched": len(snippets) > 0,
        "query": q,
        "count": len(snippets),
        "snippets": snippets,
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
):
    """Raw search endpoint with more results."""
    snippets = semantic_search(q, top_k)
    return {
        "query": q,
        "count": len(snippets),
        "results": snippets,
    }
