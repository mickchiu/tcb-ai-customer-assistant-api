# ask_api.py
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "tcb_ai_knowledge_v5.json"

app = FastAPI(
    title="TCB AI Customer Assistant API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    debug: bool = False


knowledge: List[Dict[str, Any]] = []


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_knowledge():
    global knowledge

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"找不到知識庫檔案：{KNOWLEDGE_FILE}")

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("知識庫格式錯誤，最外層必須是 list")

    knowledge = data


@app.on_event("startup")
def startup():
    load_knowledge()


def tokenize(question: str):
    question = clean_text(question)

    words = []

    # 中文 2~4 字關鍵字
    zh_parts = re.findall(r"[\u4e00-\u9fff]+", question)
    for part in zh_parts:
        if len(part) >= 2:
            words.append(part)
        for n in [2, 3, 4]:
            for i in range(len(part) - n + 1):
                words.append(part[i:i+n])

    # 英文數字
    words.extend(re.findall(r"[a-zA-Z0-9]+", question.lower()))

    stopwords = {
        "請問", "怎麼", "如何", "可以", "是否", "我要",
        "合庫", "合作金庫", "銀行", "的", "是", "嗎"
    }

    result = []
    for w in words:
        if w and w not in stopwords and w not in result:
            result.append(w)

    return result


def build_search_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("title", ""),
        item.get("intent", ""),
        item.get("page_type", ""),
        item.get("content", ""),
        item.get("faq_question", ""),
        item.get("faq_answer", ""),
    ]
    return clean_text(" ".join(parts)).lower()


def score_item(item: Dict[str, Any], question: str, terms: List[str]) -> float:
    search_text = build_search_text(item)

    title = clean_text(item.get("title")).lower()
    faq_question = clean_text(item.get("faq_question")).lower()
    faq_answer = clean_text(item.get("faq_answer")).lower()
    page_type = clean_text(item.get("page_type")).lower()

    score = 0.0

    q = clean_text(question).lower()

    if faq_question:
        if q in faq_question:
            score += 20
        if faq_question in q:
            score += 15

    for term in terms:
        term = term.lower()

        if term in faq_question:
            score += 8
        if term in title:
            score += 5
        if term in faq_answer:
            score += 4
        if term in search_text:
            score += 2

    if page_type == "faq":
        score *= 1.3

    if not item.get("source_url"):
        score *= 0.5

    return score


def search_knowledge(question: str, top_k: int = 5):
    terms = tokenize(question)

    results = []

    for item in knowledge:
        score = score_item(item, question, terms)
        if score > 0:
            results.append((item, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def make_answer(item: Dict[str, Any]) -> str:
    title = clean_text(item.get("title"))
    source_url = clean_text(item.get("source_url"))
    page_type = clean_text(item.get("page_type"))

    faq_question = clean_text(item.get("faq_question"))
    faq_answer = clean_text(item.get("faq_answer"))
    content = clean_text(item.get("content"))

    if page_type == "faq" and faq_answer:
        answer = faq_answer
        if faq_question:
            final = f"根據合庫官網資料，關於「{faq_question}」：\n\n{answer}"
        else:
            final = f"根據合庫官網 FAQ 資料：\n\n{answer}"
    else:
        answer = content[:1200]
        final = f"根據合庫官網「{title}」頁面資料：\n\n{answer}"

    if source_url:
        final += f"\n\n資料來源：{source_url}"

    return final


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TCB AI Customer Assistant API",
        "knowledge_file": str(KNOWLEDGE_FILE.name),
        "knowledge_count": len(knowledge),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "knowledge_file_exists": KNOWLEDGE_FILE.exists(),
        "knowledge_count": len(knowledge)
    }


@app.post("/ask")
def ask_post(req: AskRequest):
    results = search_knowledge(req.question, req.top_k)

    if not results:
        return {
            "question": req.question,
            "answer": "目前知識庫沒有找到足夠明確的答案，建議請客戶補充問題或轉人工確認。",
            "confidence": 0,
            "sources": []
        }

    top_item, top_score = results[0]

    answer = make_answer(top_item)

    sources = []
    for item, score in results:
        sources.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "page_type": item.get("page_type"),
            "intent": item.get("intent"),
            "source_url": item.get("source_url"),
            "score": round(score, 2),
            "preview": clean_text(item.get("faq_answer") or item.get("content"))[:150]
        })

    return {
        "question": req.question,
        "answer": answer,
        "confidence": round(min(top_score / 50, 1), 2),
        "sources": sources,
        "debug": {
            "terms": tokenize(req.question),
            "top_score": top_score
        } if req.debug else None
    }


@app.get("/ask")
def ask_get(
    q: str = Query(..., description="問題，例如：信用卡掛失怎麼辦"),
    top_k: int = Query(5, ge=1, le=10),
    debug: bool = Query(False)
):
    req = AskRequest(question=q, top_k=top_k, debug=debug)
    return ask_post(req)


@app.get("/search")
def search_get(
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=20)
):
    results = search_knowledge(q, top_k)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "page_type": item.get("page_type"),
                "intent": item.get("intent"),
                "source_url": item.get("source_url"),
                "score": round(score, 2),
                "faq_question": item.get("faq_question"),
                "preview": clean_text(item.get("faq_answer") or item.get("content"))[:200]
            }
            for item, score in results
        ]
    }