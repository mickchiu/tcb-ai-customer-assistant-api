# -*- coding: utf-8 -*-
# ask_api.py  v6.0.0 — TCB AI 客服知識檢索 API（完整版）
# 改動摘要：
#   1. VERSION 升至 6.0.0
#   2. 以 lifespan 取代已棄用的 @app.on_event("startup")
#   3. load_knowledge() 額外清洗 keywords / category / last_updated
#   4. build_search_text() 納入 keywords（×2 權重）
#   5. shorten_answer() 預設 max_len 由 380 → 500
#   6. KNOWLEDGE_FILE 改指 tcb_ai_knowledge_v6.json

import json, re, math, os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── 常數 ─────────────────────────────────────────────
VERSION = "6.0.0"
KNOWLEDGE_FILE = "tcb_ai_knowledge_v6.json"
SIM_THRESHOLD = 0.08
RISK_THRESHOLD = 0.30
TOP_K_DEFAULT = 5

INTENT_MAP = {
    "信用卡": "credit_card",
    "存款": "deposit_exchange",
    "外匯": "deposit_exchange",
    "匯率": "deposit_exchange",
    "定存": "deposit_exchange",
    "活存": "deposit_exchange",
    "數位": "digital_banking",
    "網銀": "digital_banking",
    "網路銀行": "digital_banking",
    "行動銀行": "digital_banking",
    "eATM": "digital_banking",
    "貸款": "loan",
    "房貸": "loan",
    "信貸": "loan",
    "理財": "wealth_management",
    "基金": "wealth_management",
    "保險": "insurance",
    "親子": "wealth_management",
}

RISKY_KEYWORDS = [
    "密碼", "帳號", "身分證", "個資", "詐騙", "盜刷",
    "掛失", "停卡", "凍結", "解鎖", "OTP",
]

# ── 全域變數（啟動時填入）──────────────────────────────
knowledge: List[dict] = []
corpus: List[str] = []
vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix = None


# ── 工具函式 ──────────────────────────────────────────
def clean_text(text: str) -> str:
    """移除 HTML 標籤、多餘空白"""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_intent(query: str) -> Optional[str]:
    for kw, intent in INTENT_MAP.items():
        if kw in query:
            return intent
    return None


def extract_phones(text: str) -> List[str]:
    return re.findall(r"[\d\-()]{7,}", text)


def shorten_answer(text: str, max_len: int = 500) -> str:
    """截斷過長回答，保留完整句子"""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # 往回找到最後一個句號、問號或換行
    for ch in ("。", "！", "？", "\n"):
        pos = cut.rfind(ch)
        if pos > max_len // 2:
            return cut[: pos + 1]
    return cut + "…"


def risk_guard(query: str) -> bool:
    return any(kw in query for kw in RISKY_KEYWORDS)


# ── 知識載入 ──────────────────────────────────────────
def load_knowledge():
    global knowledge, corpus, vectorizer, tfidf_matrix

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 清洗
    for item in raw:
        for key in (
            "content", "title", "faq_question", "faq_answer",
            "keywords", "category", "last_updated",
        ):
            if key in item and isinstance(item[key], str):
                item[key] = clean_text(item[key])

    knowledge = raw
    corpus = [build_search_text(item) for item in knowledge]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=80000
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(f"[v{VERSION}] 載入 {len(knowledge)} 筆知識，TF-IDF 維度 {tfidf_matrix.shape}")


def build_search_text(item: dict) -> str:
    """將各欄位組合成搜尋用文本，重要欄位加權"""
    title = item.get("title", "")
    content = item.get("content", "")
    faq_q = item.get("faq_question", "")
    faq_a = item.get("faq_answer", "")
    keywords = item.get("keywords", "")
    # title ×3、faq_question ×3、keywords ×2、其餘 ×1
    parts = [title] * 3 + [faq_q] * 3 + [keywords] * 2 + [content, faq_a]
    return " ".join(p for p in parts if p)


# ── 語意搜尋 ──────────────────────────────────────────
def semantic_search(query: str, top_k: int = TOP_K_DEFAULT,
                    intent_filter: Optional[str] = None) -> List[dict]:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    scored = []
    for idx, score in enumerate(sims):
        if score < SIM_THRESHOLD:
            continue
        item = knowledge[idx]
        # intent 篩選
        if intent_filter and item.get("intent") != intent_filter:
            continue
        # priority 加分
        priority = item.get("search_priority", "normal")
        bonus = {"high": 0.05, "normal": 0.0, "low": -0.05}.get(priority, 0.0)
        scored.append((idx, score + bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in scored[:top_k]:
        entry = dict(knowledge[idx])
        entry["score"] = round(float(score), 4)
        results.append(entry)
    return results


# ── 回答組裝 ──────────────────────────────────────────
def build_sources(results: List[dict]) -> List[dict]:
    sources = []
    for r in results:
        sources.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "url": r.get("source_url"),
            "score": r.get("score"),
        })
    return sources


def make_risk_answer(query: str) -> dict:
    return {
        "answer": (
            "您的問題涉及帳戶安全相關事項，為保障您的權益，"
            "建議您撥打合庫客服專線 (04)2227-3131 或 0800-033175，"
            "由專人為您處理。"
        ),
        "sources": [],
        "intent": detect_intent(query),
        "risk_flag": True,
        "version": VERSION,
    }


def make_customer_answer(query: str, results: List[dict]) -> dict:
    if not results:
        return no_answer(query)

    best = results[0]
    # 優先使用 FAQ
    if best.get("faq_answer"):
        answer_text = best["faq_answer"]
    else:
        answer_text = best.get("content", "")

    answer_text = shorten_answer(answer_text)
    phones = extract_phones(answer_text)

    return {
        "answer": answer_text,
        "sources": build_sources(results),
        "intent": best.get("intent"),
        "risk_flag": False,
        "phones": phones,
        "version": VERSION,
    }


def compact_answer(query: str, results: List[dict]) -> str:
    """純文字回覆（給 /ask_text）"""
    if not results:
        return f"很抱歉，目前找不到與「{query}」相關的資訊。建議撥打客服專線 0800-033175。"
    best = results[0]
    if best.get("faq_answer"):
        return shorten_answer(best["faq_answer"])
    return shorten_answer(best.get("content", ""))


def no_answer(query: str) -> dict:
    return {
        "answer": f"很抱歉，目前找不到與「{query}」相關的資訊。建議您撥打客服專線 0800-033175 或 (04)2227-3131。",
        "sources": [],
        "intent": detect_intent(query),
        "risk_flag": False,
        "version": VERSION,
    }


# ── FastAPI lifespan ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_knowledge()
    yield
    # shutdown（如有需要可在此釋放資源）


app = FastAPI(
    title="TCB AI 客服知識 API",
    version=VERSION,
    lifespan=lifespan,
)


# ── Request Model ─────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT
    intent: Optional[str] = None


# ── API 端點 ──────────────────────────────────────────
@app.post("/ask")
async def ask_post(req: AskRequest):
    query = req.query.strip()
    if not query:
        return JSONResponse({"error": "query 不可為空"}, status_code=400)

    if risk_guard(query):
        return make_risk_answer(query)

    intent = req.intent or detect_intent(query)
    results = semantic_search(query, top_k=req.top_k, intent_filter=intent)
    return make_customer_answer(query, results)


@app.get("/ask")
async def ask_get(
    q: str = Query(..., min_length=1),
    top_k: int = Query(TOP_K_DEFAULT, ge=1, le=20),
    intent: Optional[str] = Query(None),
):
    query = q.strip()
    if risk_guard(query):
        return make_risk_answer(query)

    intent_val = intent or detect_intent(query)
    results = semantic_search(query, top_k=top_k, intent_filter=intent_val)
    return make_customer_answer(query, results)


@app.get("/ask_text")
async def ask_text(
    q: str = Query(..., min_length=1),
    top_k: int = Query(TOP_K_DEFAULT, ge=1, le=20),
    intent: Optional[str] = Query(None),
):
    query = q.strip()
    if risk_guard(query):
        return make_risk_answer(query)["answer"]

    intent_val = intent or detect_intent(query)
    results = semantic_search(query, top_k=top_k, intent_filter=intent_val)
    return compact_answer(query, results)


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(TOP_K_DEFAULT, ge=1, le=20),
    intent: Optional[str] = Query(None),
):
    query = q.strip()
    intent_val = intent or detect_intent(query)
    results = semantic_search(query, top_k=top_k, intent_filter=intent_val)
    return {"query": query, "intent": intent_val, "results": results}


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <html><body>
    <h2>TCB AI 客服知識 API v{VERSION}</h2>
    <p>知識庫筆數：{len(knowledge)}</p>
    <ul>
      <li>POST /ask — JSON 查詢</li>
      <li>GET  /ask?q=... — 快速查詢</li>
      <li>GET  /ask_text?q=... — 純文字回覆</li>
      <li>GET  /search?q=... — 原始搜尋結果</li>
      <li>GET  /health — 健康檢查</li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": VERSION,
        "knowledge_count": len(knowledge),
        "tfidf_shape": list(tfidf_matrix.shape) if tfidf_matrix is not None else None,
    }


# ── 本機執行 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ask_api:app", host="0.0.0.0", port=8000, reload=True)
