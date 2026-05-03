# ask_api.py
# -*- coding: utf-8 -*-
"""
TCB AI Customer Assistant API

用途：
- 讀取 tcb_ai_knowledge_v5.json
- 提供 /ask 給 Copilot Studio / Power Automate / Custom Connector 呼叫
- FAQ 精準優先
- 避免 sources 混入太多不相關資料
- 回傳 answer / confidence / sources
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "tcb_ai_knowledge_v5.json"

app = FastAPI(
    title="TCB AI Customer Assistant API",
    version="1.1.0"
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


STOPWORDS = {
    "請問", "怎麼", "如何", "可以", "是否", "我要", "想問",
    "合庫", "合作金庫", "銀行", "本行", "的", "是", "嗎", "呢",
    "一下", "辦理", "申請", "相關", "服務"
}

SYNONYMS = {
    "掛失": ["遺失", "不見", "被竊", "被偷", "掉了", "補發"],
    "信用卡": ["卡片", "國際信用卡", "持卡", "刷卡"],
    "金融卡": ["visa金融卡", "combo卡", "提款卡"],
    "繳稅": ["牌照稅", "地價稅", "房屋稅", "所得稅", "網路繳稅"],
    "預借現金": ["借現金", "現金", "預借"],
    "額度": ["信用額度", "臨時提高", "調高額度"],
    "帳單": ["帳款", "消費明細", "繳款"],
    "客服": ["電話", "專線", "客服中心"],
    "機場接送": ["接機", "送機", "機場", "肯驛"],
}


def clean_text(text: Any) -> str:
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

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        item = dict(item)
        item["title"] = clean_text(item.get("title"))
        item["intent"] = clean_text(item.get("intent"))
        item["page_type"] = clean_text(item.get("page_type"))
        item["source_url"] = clean_text(item.get("source_url"))
        item["content"] = clean_text(item.get("content"))
        item["faq_question"] = clean_text(item.get("faq_question"))
        item["faq_answer"] = clean_text(item.get("faq_answer"))

        cleaned.append(item)

    knowledge = cleaned


@app.on_event("startup")
def startup():
    load_knowledge()


def tokenize(question: str) -> List[str]:
    question = clean_text(question).lower()
    words: List[str] = []

    words.extend(re.findall(r"[a-zA-Z0-9]+", question))

    zh_parts = re.findall(r"[\u4e00-\u9fff]+", question)
    for part in zh_parts:
        if len(part) >= 2:
            words.append(part)

        for n in [2, 3, 4]:
            if len(part) >= n:
                for i in range(len(part) - n + 1):
                    words.append(part[i:i+n])

    expanded = list(words)
    for key, values in SYNONYMS.items():
        if key in question or any(v in question for v in values):
            expanded.append(key)
            expanded.extend(values)

    result = []
    for w in expanded:
        w = clean_text(w).lower()
        if w and w not in STOPWORDS and w not in result:
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

    q = clean_text(question).lower()
    score = 0.0

    if faq_question:
        if q == faq_question:
            score += 100
        elif q in faq_question or faq_question in q:
            score += 60

    important_terms = [
        "掛失", "遺失", "不見", "被竊",
        "預借現金", "繳稅", "機場接送",
        "帳單", "額度", "客服", "電話"
    ]

    for term in terms:
        term = term.lower()

        weight = 1.0
        if term in important_terms:
            weight = 2.0

        if term in faq_question:
            score += 12 * weight
        if term in title:
            score += 5 * weight
        if term in faq_answer:
            score += 4 * weight
        if term in search_text:
            score += 1.5 * weight

    if page_type == "faq":
        score *= 1.35

    if not item.get("source_url"):
        score *= 0.5

    return score


def search_knowledge(question: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
    terms = tokenize(question)
    results: List[Tuple[Dict[str, Any], float]] = []

    for item in knowledge:
        score = score_item(item, question, terms)
        if score > 0:
            results.append((item, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def build_sources(results: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    if not results:
        return []

    top_score = results[0][1]
    filtered = []

    for item, score in results:
        if score >= top_score * 0.55:
            filtered.append((item, score))

    if not filtered:
        filtered = [results[0]]

    sources = []
    for item, score in filtered[:3]:
        sources.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "page_type": item.get("page_type"),
            "intent": item.get("intent"),
            "source_url": item.get("source_url"),
            "score": round(score, 2),
            "faq_question": item.get("faq_question"),
            "preview": clean_text(item.get("faq_answer") or item.get("content"))[:200]
        })

    return sources


def make_customer_answer(item: Dict[str, Any]) -> str:
    title = clean_text(item.get("title"))
    source_url = clean_text(item.get("source_url"))
    page_type = clean_text(item.get("page_type"))

    faq_question = clean_text(item.get("faq_question"))
    faq_answer = clean_text(item.get("faq_answer"))
    content = clean_text(item.get("content"))

    raw_answer = faq_answer if (page_type == "faq" and faq_answer) else content
    raw_answer = raw_answer[:1500]

    # ===== 分析內容 =====
    is_lost = any(k in (faq_question + raw_answer) for k in ["掛失", "遺失", "被竊"])
    
    # 抓電話
    phones = re.findall(r"(0\d{1,3}-?\d{6,8}|0800-?\d{3}-?\d{3})", raw_answer)
    phone_text = " / ".join(set(phones)) if phones else "請洽官方客服"

    # ===== 組客服格式 =====
    if is_lost:
        answer = f"""【處理方式】
信用卡或金融卡遺失時，請立即辦理掛失，以避免被冒用。

【步驟】
1. 立即撥打客服專線辦理掛失
2. 後續至原發卡分行補辦書面手續
3. 申請補發新卡

【注意事項】
- 掛失後即可停止卡片使用，降低風險
- 掛失前可能有自負額（依銀行規定）
- 掛失後通常會收取手續費

【客服電話】
{phone_text}

【詳細說明】
{raw_answer}

資料來源：{source_url}
"""
    else:
        answer = f"""【處理方式】
{title or "請參考以下說明"}

【步驟】
請依下列方式辦理：

【詳細說明】
{raw_answer}

【客服電話】
{phone_text}

資料來源：{source_url}
"""

    return answer.strip()


def no_answer(question: str) -> str:
    return (
        "目前知識庫沒有找到足夠明確的答案，為避免提供錯誤資訊，"
        "建議請客戶補充業務類型、卡別或申辦項目，或轉由人工客服確認。"
    )


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
    question = clean_text(req.question)
    results = search_knowledge(question, req.top_k)

    if not results:
        return {
            "question": question,
            "answer": no_answer(question),
            "confidence": 0,
            "sources": []
        }

    top_item, top_score = results[0]
    confidence = round(min(top_score / 80, 1), 2)

    if confidence < 0.25:
        return {
            "question": question,
            "answer": no_answer(question),
            "confidence": confidence,
            "sources": build_sources(results)
        }

    answer = make_customer_answer(top_item)
    sources = build_sources(results)

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "debug": {
            "terms": tokenize(question),
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


@app.get("/ask_text")
def ask_text(
    q: str = Query(..., description="問題，例如：信用卡掛失怎麼辦"),
    top_k: int = Query(5, ge=1, le=10)
):
    req = AskRequest(question=q, top_k=top_k, debug=False)
    result = ask_post(req)
    return result["answer"]


@app.get("/search")
def search_get(
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=20)
):
    results = search_knowledge(q, top_k)

    return {
        "query": q,
        "count": len(results),
        "results": build_sources(results)
    }
