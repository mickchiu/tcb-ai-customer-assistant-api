# ask_api.py
# -*- coding: utf-8 -*-

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
    version="3.0.0"
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
    "一下", "辦理", "申請", "相關", "服務", "問題", "資料"
}


SYNONYMS = {
    "掛失": ["遺失", "不見", "被竊", "被偷", "掉了", "補發"],
    "信用卡": ["卡片", "國際信用卡", "持卡", "刷卡"],
    "金融卡": ["visa金融卡", "combo卡", "提款卡"],
    "開戶": ["帳戶", "存款", "數位帳戶", "預約開戶"],
    "貸款": ["房貸", "信貸", "信用貸款", "房屋貸款"],
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
        item["id"] = clean_text(item.get("id"))
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


def detect_intent(question: str) -> str:
    q = clean_text(question).lower()

    if any(k in q for k in ["信用卡", "卡片", "掛失", "刷卡", "預借現金", "額度", "帳單"]):
        return "信用卡"

    if any(k in q for k in ["開戶", "帳戶", "存款", "數位帳戶"]):
        return "開戶"

    if any(k in q for k in ["貸款", "房貸", "信貸", "信用貸款"]):
        return "貸款"

    if any(k in q for k in ["繳稅", "牌照稅", "地價稅", "房屋稅", "所得稅"]):
        return "繳稅"

    if any(k in q for k in ["機場接送", "接機", "送機"]):
        return "信用卡優惠服務"

    return "其他"


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
            score += 120
        elif q in faq_question or faq_question in q:
            score += 80

    important_terms = {
        "掛失", "遺失", "不見", "被竊", "被偷",
        "信用卡", "金融卡", "預借現金", "繳稅",
        "機場接送", "帳單", "額度", "客服", "電話",
        "開戶", "貸款", "房貸", "信貸"
    }

    for term in terms:
        weight = 2.2 if term in important_terms else 1.0

        if term in faq_question:
            score += 14 * weight
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


def shorten_answer(text: str, max_len: int = 450) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def extract_phones(text: str) -> str:
    raw = re.findall(
        r"(?:\(?0\d{1,3}\)?-?\d{3,4}-?\d{3,4}|0800-?\d{3}-?\d{3}|886-?\d-?\d{4}-?\d{4})",
        text
    )

    normalized = []

    for p in raw:
        p = p.replace("(", "").replace(")", "").replace(" ", "")

        if p == "0800033175":
            p = "0800-033-175"

        if p == "0800-033175":
            p = "0800-033-175"

        if p == "04-22273131":
            p = "04-2227-3131"

        if p == "0422273131":
            p = "04-2227-3131"
            
        if p == "042227-3131":
            p = "04-2227-3131"

        if p not in normalized:
            normalized.append(p)

    preferred = []

    for p in ["0800-033-175", "04-2227-3131"]:
        if p in normalized:
            preferred.append(p)

    for p in normalized:
        if p not in preferred:
            preferred.append(p)

    if preferred:
        return " / ".join(preferred[:2])

    return "請洽合庫官方客服或鄰近分行確認。"


def risk_guard(question: str, confidence: float) -> bool:
    q = clean_text(question).lower()

    risky_keywords = [
        "核准", "一定過", "保證", "個人資料", "查帳", "餘額",
        "我的帳戶", "我的信用卡", "我的貸款", "我的額度",
        "利率多少", "可以借多少", "可貸多少", "審核結果",
        "身分證", "密碼", "otp", "驗證碼"
    ]

    if confidence < 0.4:
        return True

    if any(k in q for k in risky_keywords):
        return True

    return False


def make_risk_answer(question: str, confidence: float, sources: List[Dict[str, Any]]) -> str:
    intent = detect_intent(question)

    source_url = ""
    if sources:
        source_url = sources[0].get("source_url") or ""

    answer = f"""【業務類型】
{intent}

【處理方式】
這個問題可能涉及個人條件、帳務資料或即時審核結果。為避免提供錯誤資訊，建議由人工客服或分行協助確認。

【步驟】
1. 請準備欲詢問的業務類型與基本資料
2. 聯繫合庫客服或洽鄰近分行
3. 若涉及個人帳務、額度、核准或利率，請以銀行查詢結果為準

【注意事項】
- 請勿在聊天中提供密碼、OTP、驗證碼或完整身分證字號
- 個人帳務、信用額度、貸款核准與利率條件需由銀行系統或人工確認
- 本回覆僅作一般資訊引導，不代表最終審核或交易結果

【客服電話】
0800-033-175 / 04-2227-3131"""

    if source_url:
        answer += f"""

【資料來源】
{source_url}"""

    return answer.strip()


def make_customer_answer(item: Dict[str, Any], question: str) -> str:
    intent = detect_intent(question)

    title = clean_text(item.get("title"))
    source_url = clean_text(item.get("source_url"))
    page_type = clean_text(item.get("page_type"))

    faq_question = clean_text(item.get("faq_question"))
    faq_answer = clean_text(item.get("faq_answer"))
    content = clean_text(item.get("content"))

    raw_answer = faq_answer if (page_type == "faq" and faq_answer) else content
    short_answer = shorten_answer(raw_answer)
    phone_text = extract_phones(raw_answer)

    question_context = faq_question or title or "相關問題"

    is_lost = any(k in (question + faq_question + raw_answer) for k in ["掛失", "遺失", "被竊", "不見", "掉了"])
    is_cash_advance = any(k in (question + faq_question + raw_answer) for k in ["預借現金", "借現金"])
    is_tax = any(k in (question + faq_question + raw_answer) for k in ["繳稅", "牌照稅", "地價稅", "房屋稅", "所得稅"])
    is_airport = any(k in (question + faq_question + raw_answer) for k in ["機場接送", "接機", "送機"])

    if is_lost:
        process = "信用卡或金融卡遺失時，請立即辦理掛失，以避免卡片遭冒用。"
        steps = (
            "1. 立即撥打客服專線辦理口頭掛失\n"
            "2. 依銀行規定至分行補辦書面手續\n"
            "3. 如需繼續使用，申請補發新卡"
        )
        notes = (
            "- 掛失後可降低卡片遭冒用風險\n"
            "- 掛失前如已遭冒用，仍可能依規定負擔自付額\n"
            "- 掛失或補發可能產生手續費，實際費用請依官網或分行說明為準"
        )

    elif is_cash_advance:
        process = "若需要小額資金，可依合庫信用卡預借現金規定辦理。"
        steps = (
            "1. 確認信用卡是否已申請預借現金密碼\n"
            "2. 於可支援的 ATM 或指定櫃台辦理\n"
            "3. 留意每日限額、次數與手續費"
        )
        notes = (
            "- 預借現金通常會收取手續費\n"
            "- 是否可辦理與額度限制，仍依卡片種類與銀行規定為準\n"
            "- 若涉及個人額度，建議洽客服確認"
        )

    elif is_tax:
        process = "可依合庫官網提供的信用卡繳稅方式辦理。"
        steps = (
            "1. 確認欲繳納的稅目\n"
            "2. 依官網提供的繳稅平台或連結操作\n"
            "3. 留意每筆手續費與公告規定"
        )
        notes = (
            "- 不同稅目可能有不同手續費或限制\n"
            "- 實際可繳項目與費用請以官網公告為準"
        )

    elif is_airport:
        process = "符合指定卡別與消費條件者，可依活動辦法預約機場接送服務。"
        steps = (
            "1. 確認持有卡別是否符合活動資格\n"
            "2. 確認指定期間內是否完成符合條件的刷卡消費\n"
            "3. 依官網公告的預約方式辦理"
        )
        notes = (
            "- 機場接送通常有預約期限、年度次數與服務區域限制\n"
            "- 連續假期或旅遊旺季建議提早預約\n"
            "- 實際資格、趟次與費用請以官網公告為準"
        )

    else:
        process = f"以下為合庫官網針對「{question_context}」提供的說明。"
        steps = "請依合庫官網或業務單位公告方式辦理；如涉及個人資料、帳務狀態或資格條件，建議洽客服或分行確認。"
        notes = (
            "- 各項業務可能依身分、產品別、申請方式或最新公告而不同\n"
            "- 若涉及費用、資格或時效，建議以官網與分行說明為準"
        )

    answer = f"""【業務類型】
{intent}

【處理方式】
{process}

【步驟】
{steps}

【注意事項】
{notes}

【客服電話】
{phone_text}

【詳細說明】
{short_answer}

【資料來源】
{source_url}"""

    return answer.strip()


def no_answer(question: str) -> str:
    return (
        "【處理方式】\n"
        "目前知識庫沒有找到足夠明確的答案，為避免提供錯誤資訊，建議轉由人工客服確認。\n\n"
        "【步驟】\n"
        "1. 請補充業務類型，例如信用卡、開戶、貸款或繳稅\n"
        "2. 若涉及個人帳務、身分資料或即時狀態，請以客服或分行查詢結果為準\n\n"
        "【客服電話】\n"
        "0800-033-175 / 04-2227-3131"
    )


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TCB AI Customer Assistant API",
        "version": "3.0.0",
        "knowledge_file": str(KNOWLEDGE_FILE.name),
        "knowledge_count": len(knowledge),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
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
            "intent": detect_intent(question),
            "sources": []
        }

    top_item, top_score = results[0]
    confidence = round(min(top_score / 80, 1), 2)
    sources = build_sources(results)

    if risk_guard(question, confidence):
        return {
            "question": question,
            "answer": make_risk_answer(question, confidence, sources),
            "confidence": confidence,
            "intent": detect_intent(question),
            "sources": sources
        }

    answer = make_customer_answer(top_item, question)

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "intent": detect_intent(question),
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


@app.get("/ask_text", response_class=PlainTextResponse)
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
