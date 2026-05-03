# ask_api.py
# -*- coding: utf-8 -*-
# ===================================================================
#  TCB AI Customer Assistant API — v4.1.0（輕量 RAG：TF-IDF + jieba）
#  ‧ 使用 scikit-learn TF-IDF + jieba 中文斷詞做語意搜尋
#  ‧ 不需要 sentence-transformers、不需要 PyTorch
#  ‧ 不需要 OpenAI API Key
#  ‧ Render 免費方案可跑（RAM < 100MB）
#  ‧ 保持與 v3 完全一致的 API 介面，Copilot 端零修改
# ===================================================================

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# =====================  路徑與常數  =====================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "tcb_ai_knowledge_v5.json"

TOP_K_DEFAULT  = 5
SIM_THRESHOLD  = 0.08        # TF-IDF cosine 閾值（比 embedding 低很多是正常的）
RISK_THRESHOLD = 0.10        # 低於此 → 風險題

VERSION = "4.1.0"

# =====================  FastAPI 初始化  =====================

app = FastAPI(
    title="TCB AI Customer Assistant API (RAG Lite)",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================  Request Model  =====================

class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT
    debug: bool = False
    compact: bool = True

# =====================  全域變數  =====================

knowledge: List[Dict[str, Any]] = []
tfidf_vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix = None   # scipy sparse matrix

# =====================  Intent 中文對照  =====================

INTENT_MAP = {
    "credit_card":      "信用卡",
    "foreign_exchange":  "存款/外匯",
    "digital_banking":   "數位金融",
    "insurance_trust":   "保險/信託",
    "loan":              "貸款",
    "deposit_account":   "存款帳戶",
    "general":           "一般業務",
}

# =====================  風險關鍵字  =====================

RISKY_KEYWORDS = [
    "核准", "一定過", "保證", "個人資料", "查帳", "餘額",
    "我的帳戶", "我的信用卡", "我的貸款", "我的額度",
    "利率多少", "可以借多少", "可貸多少", "審核結果",
    "身分證", "密碼", "otp", "驗證碼",
]

# =====================  jieba 自訂詞典  =====================

CUSTOM_WORDS = [
    "信用卡", "金融卡", "預借現金", "機場接送", "機場貴賓室",
    "掛失", "補發", "數位帳戶", "台灣Pay", "網路銀行",
    "行動網銀", "定期存款", "活期存款", "綜合存款",
    "信用貸款", "房屋貸款", "以房養老", "微型創業",
    "留學生貸款", "黃金存摺", "外幣現鈔", "牌照稅",
    "地價稅", "房屋稅", "所得稅", "繳稅", "手續費",
    "年費", "循環利率", "違約金", "紅利積點",
    "現金回饋", "帳單分期", "合庫", "合作金庫",
    "未成年", "開戶", "數位存款", "保管箱", "匯款",
    "VISA", "Combo", "JCB", "Mastercard",
]

def init_jieba():
    """初始化 jieba 並加入銀行專用詞"""
    for word in CUSTOM_WORDS:
        jieba.add_word(word)
    # 預熱斷詞引擎
    jieba.lcut("合作金庫信用卡掛失")
    print("[啟動] jieba 斷詞引擎初始化完成")


def jieba_tokenize(text: str) -> str:
    """用 jieba 斷詞，回傳空格分隔的字串（給 TF-IDF 用）"""
    words = jieba.lcut(text)
    return " ".join(words)

# =====================  工具函式  =====================

def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_search_text(item: Dict[str, Any]) -> str:
    """
    組合搜尋用文字：faq_question + title + content 前 600 字
    用 jieba 斷詞後回傳
    """
    parts = []
    faq_q = clean_text(item.get("faq_question"))
    if faq_q:
        # FAQ 問題重複 3 次以提高權重
        parts.extend([faq_q] * 3)
    title = clean_text(item.get("title", ""))
    if title:
        parts.extend([title] * 2)
    content = clean_text(item.get("content", ""))
    if content:
        parts.append(content[:600])
    raw = " ".join(parts)
    return jieba_tokenize(raw)


def detect_intent(question: str) -> str:
    q = clean_text(question).lower()
    if any(k in q for k in ["金融卡", "visa金融卡", "combo卡", "提款卡"]):
        return "金融卡"
    if any(k in q for k in ["開戶", "帳戶", "存款", "數位帳戶", "未成年", "未成年人"]):
        return "開戶"
    if any(k in q for k in ["貸款", "房貸", "信貸", "信用貸款"]):
        return "貸款"
    if any(k in q for k in ["繳稅", "牌照稅", "地價稅", "房屋稅", "所得稅"]):
        return "繳稅"
    if any(k in q for k in ["機場接送", "接機", "送機"]):
        return "信用卡優惠服務"
    if any(k in q for k in ["信用卡", "卡片", "掛失", "刷卡", "預借現金", "額度", "帳單"]):
        return "信用卡"
    return "其他"


def extract_phones(text: str) -> str:
    raw = re.findall(
        r"(?:\(?0\d{1,3}\)?-?\d{3,4}-?\d{3,4}|0800-?\d{3}-?\d{3}|886-?\d-?\d{4}-?\d{4})",
        text,
    )
    normalized = []
    for p in raw:
        digits = re.sub(r"\D", "", p)
        p = p.replace("(", "").replace(")", "").replace(" ", "")
        if digits == "0800033175":
            p = "0800-033-175"
        elif digits == "0422273131":
            p = "04-2227-3131"
        elif digits == "886422273131":
            p = "886-4-2227-3131"
        if p not in normalized:
            normalized.append(p)

    preferred = []
    for p in ["0800-033-175", "04-2227-3131", "886-4-2227-3131"]:
        if p in normalized:
            preferred.append(p)
    for p in normalized:
        if p not in preferred:
            preferred.append(p)

    if preferred:
        return " / ".join(preferred[:2])
    return "0800-033-175 / 04-2227-3131"


def shorten_answer(text: str, max_len: int = 380) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def risk_guard(question: str, confidence: float) -> bool:
    q = clean_text(question).lower()
    if confidence < RISK_THRESHOLD:
        return True
    if any(k in q for k in RISKY_KEYWORDS):
        return True
    return False

# =====================  TF-IDF 搜尋核心  =====================

def semantic_search(question: str, top_k: int = TOP_K_DEFAULT) -> List[Tuple[Dict[str, Any], float]]:
    """
    TF-IDF 語意搜尋：
    1. 將問題用 jieba 斷詞
    2. 轉成 TF-IDF 向量
    3. 與所有文件做 cosine similarity
    4. 取 top_k
    """
    global tfidf_vectorizer, tfidf_matrix, knowledge

    q_tokenized = jieba_tokenize(clean_text(question))
    q_vec = tfidf_vectorizer.transform([q_tokenized])
    scores = sk_cosine(q_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sim = float(scores[idx])
        if sim >= SIM_THRESHOLD:
            results.append((knowledge[idx], sim))

    return results

# =====================  來源與回答組裝  =====================

def build_sources(results: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    if not results:
        return []

    top_score = results[0][1]
    filtered = [(item, s) for item, s in results if s >= top_score * 0.55]
    if not filtered:
        filtered = [results[0]]

    sources = []
    for item, score in filtered[:3]:
        sources.append({
            "id":           item.get("id"),
            "title":        item.get("title"),
            "page_type":    item.get("page_type"),
            "intent":       item.get("intent"),
            "source_url":   item.get("source_url"),
            "score":        round(score, 4),
            "faq_question": item.get("faq_question"),
            "preview":      clean_text(
                                item.get("faq_answer") or item.get("content")
                            )[:200],
        })
    return sources


def make_risk_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    intent = detect_intent(question)
    source_url = sources[0].get("source_url", "") if sources else ""

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
        answer += f"\n\n【資料來源】\n{source_url}"

    return answer.strip()


def make_customer_answer(item: Dict[str, Any], question: str, sim_score: float) -> str:
    intent_raw = clean_text(item.get("intent"))
    intent = INTENT_MAP.get(intent_raw, detect_intent(question))

    title      = clean_text(item.get("title"))
    source_url = clean_text(item.get("source_url"))
    page_type  = clean_text(item.get("page_type"))

    faq_question = clean_text(item.get("faq_question"))
    faq_answer   = clean_text(item.get("faq_answer"))
    content      = clean_text(item.get("content"))

    raw_answer = faq_answer if (page_type == "faq" and faq_answer) else content
    detail     = shorten_answer(raw_answer)
    phone_text = extract_phones(raw_answer)

    question_context = faq_question or title or "相關問題"

    combined = question + faq_question + raw_answer
    is_lost         = any(k in combined for k in ["掛失", "遺失", "被竊", "不見", "掉了"])
    is_cash_advance = any(k in combined for k in ["預借現金", "借現金"])
    is_tax          = any(k in combined for k in ["繳稅", "牌照稅", "地價稅", "房屋稅", "所得稅"])
    is_airport      = any(k in combined for k in ["機場接送", "接機", "送機"])
    is_opening      = detect_intent(question) == "開戶"

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
    elif is_opening:
        process = f"以下為合庫官網針對「{question_context}」提供的開戶相關說明。"
        steps = (
            "1. 請確認開戶人身分與年齡條件\n"
            "2. 準備身分證明文件及銀行要求文件\n"
            "3. 若為未成年人，建議先洽分行確認是否需法定代理人陪同或提供相關文件"
        )
        notes = (
            "- 未成年人開戶通常涉及法定代理人或監護人文件要求\n"
            "- 實際所需文件可能依帳戶類型、開戶方式與分行規定不同\n"
            "- 請以合庫官網與分行最新說明為準"
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
{detail}

【資料來源】
{source_url}"""

    return answer.strip()


def compact_answer(answer: str, sources: List[Dict[str, Any]]) -> str:
    source_url = sources[0].get("source_url", "") if sources else ""

    if "【詳細說明】" in answer:
        answer = answer.split("【詳細說明】")[0].strip()

    if "【資料來源】" not in answer:
        answer += f"\n\n【資料來源】\n{source_url}"
    elif source_url and source_url not in answer:
        answer += f"\n{source_url}"

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

# =====================  知識庫載入 & TF-IDF 建置  =====================

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
        for key in ["id", "title", "intent", "page_type", "source_url",
                     "content", "faq_question", "faq_answer"]:
            item[key] = clean_text(item.get(key))
        cleaned.append(item)

    knowledge = cleaned
    print(f"[啟動] 知識庫載入完成，共 {len(knowledge)} 筆文件")


def build_tfidf_index():
    """建立 TF-IDF 向量索引"""
    global tfidf_vectorizer, tfidf_matrix

    print(f"[啟動] 正在建立 TF-IDF 索引（{len(knowledge)} 筆文件）...")

    # 組合所有文件的搜尋文字（已含 jieba 斷詞）
    doc_texts = [build_search_text(item) for item in knowledge]

    # 建立 TF-IDF 向量化器
    # analyzer="word" 因為已經用 jieba 斷好了
    # ngram_range=(1,2) 支援二元組合，例如「信用 卡」「預借 現金」
    # sublinear_tf=True 對長文件做 log 壓縮，避免長文壟斷
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)
    print(f"[啟動] TF-IDF 索引建立完成，矩陣 shape={tfidf_matrix.shape}")

# =====================  API 端點  =====================

@app.on_event("startup")
def startup():
    init_jieba()
    load_knowledge()
    build_tfidf_index()
    print(f"[啟動] TCB AI RAG Lite API v{VERSION} 啟動完成 ✓")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TCB AI Customer Assistant API (RAG Lite)",
        "version": VERSION,
        "knowledge_file": str(KNOWLEDGE_FILE.name),
        "knowledge_count": len(knowledge),
        "search_engine": "TF-IDF + jieba",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "knowledge_file_exists": KNOWLEDGE_FILE.exists(),
        "knowledge_count": len(knowledge),
        "tfidf_ready": tfidf_matrix is not None,
    }


@app.post("/ask")
def ask_post(req: AskRequest):
    question = clean_text(req.question)

    results = semantic_search(question, req.top_k)

    if not results:
        answer = no_answer(question)
        return {
            "question":    question,
            "answer":      answer,
            "full_answer": answer,
            "confidence":  0,
            "intent":      detect_intent(question),
            "sources":     [],
        }

    top_item, top_score = results[0]
    confidence = round(min(top_score / 0.35, 1.0), 2)
    sources = build_sources(results)

    if risk_guard(question, confidence):
        answer = make_risk_answer(question, sources)
    else:
        answer = make_customer_answer(top_item, question, top_score)

    display_answer = compact_answer(answer, sources) if req.compact else answer

    return {
        "question":    question,
        "answer":      display_answer,
        "full_answer": answer,
        "confidence":  confidence,
        "intent":      detect_intent(question),
        "sources":     sources,
        "debug": {
            "top_sim_score": round(top_score, 4),
            "engine": "tfidf+jieba",
        } if req.debug else None,
    }


@app.get("/ask")
def ask_get(
    q:       str  = Query(..., description="問題，例如：信用卡掛失怎麼辦"),
    top_k:   int  = Query(TOP_K_DEFAULT, ge=1, le=10),
    debug:   bool = Query(False),
    compact: bool = Query(True, description="是否回傳短版答案，Copilot 建議使用 true"),
):
    req = AskRequest(question=q, top_k=top_k, debug=debug, compact=compact)
    return ask_post(req)


@app.get("/ask_text", response_class=PlainTextResponse)
def ask_text(
    q:       str  = Query(..., description="問題，例如：信用卡掛失怎麼辦"),
    top_k:   int  = Query(TOP_K_DEFAULT, ge=1, le=10),
    compact: bool = Query(True),
):
    req = AskRequest(question=q, top_k=top_k, debug=False, compact=compact)
    result = ask_post(req)
    return result["answer"]


@app.get("/search")
def search_get(
    q:     str = Query(...),
    top_k: int = Query(TOP_K_DEFAULT, ge=1, le=20),
):
    results = semantic_search(q, top_k)
    return {
        "query":   q,
        "count":   len(results),
        "results": build_sources(results),
    }
