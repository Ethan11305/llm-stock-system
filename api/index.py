# ╔══════════════════════════════════════════════════════════════╗
# ║  LLM理財 — FastAPI 後端（Vercel serverless entry point）      ║
# ║  本檔案是 Vercel 的入口，內容與 backend/main.py 保持同步。    ║
# ╚══════════════════════════════════════════════════════════════╝

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import time

app = FastAPI(title="LLM理財 API", version="0.1.0")

# ── CORS（讓 Streamlit 可以跨 origin 呼叫）──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
#  資料結構
# ══════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    ticker: Optional[str] = None   # 例如 "2330"

class Fact(BaseModel):
    id: int
    text: str
    confidence: int   # 0–100
    source: str

class Risk(BaseModel):
    id: int
    level: str        # "high" | "medium" | "low"
    text: str
    source: str

class Evidence(BaseModel):
    id: int
    title: str
    content: str
    heat: int         # 0–100  證據熱度
    tags: list[str]

class Source(BaseModel):
    id: int
    title: str
    type: str         # "official" | "media" | "research"
    date: str
    relevance: int    # 0–100

class QueryResult(BaseModel):
    status: str       # "success" | "partial" | "rejected"
    ticker: str
    name: str
    price: str
    change: str
    change_pct: str
    query: str
    trust_score: int
    timestamp: str
    summary: str
    facts: list[Fact]
    risks: list[Risk]
    evidence: list[Evidence]
    sources: list[Source]

class HistoryItem(BaseModel):
    id: int
    query: str
    time: str
    status: str

# ══════════════════════════════════════════════════════════════
#  Mock 資料（之後換成真實 LLM 呼叫）
# ══════════════════════════════════════════════════════════════

MOCK_HISTORY: list[HistoryItem] = [
    HistoryItem(id=1, query="台積電近期法說會重點與 AI 晶片需求展望", time="14:32", status="success"),
    HistoryItem(id=2, query="聯發科 2025 下半年營收展望與天璣競爭力",  time="13:15", status="success"),
    HistoryItem(id=3, query="台灣半導體對美國關稅政策影響評估",         time="11:40", status="partial"),
    HistoryItem(id=4, query="鴻海 AI 伺服器業務佔比與輝達合作進展",     time="10:22", status="success"),
    HistoryItem(id=5, query="DRAM 價格走勢與南亞科技 2026 展望",        time="09:55", status="rejected"),
    HistoryItem(id=6, query="台灣 ETF 0050 vs 0056 長期績效比較",       time="昨天",  status="success"),
    HistoryItem(id=7, query="聯電先進製程策略與市佔率分析",             time="昨天",  status="success"),
]

def build_mock_result(query: str) -> QueryResult:
    """
    TODO: 將此函式替換為真實的 LLM / RAG 呼叫。
    """
    return QueryResult(
        status="success",
        ticker="2330",
        name="台積電",
        price="1,208",
        change="+38",
        change_pct="+3.25%",
        query=query,
        trust_score=87,
        timestamp=datetime.now().strftime("%Y-%m-%d  %H:%M"),
        summary=(
            "台積電 2026Q1 法說會上調全年美元收入成長指引至「中段 20% 以上」，"
            "AI 相關需求強勁帶動 CoWoS 先進封裝持續滿載。2nm N2 製程已進入量產且"
            "良率達標，毛利率指引維持 57–59%。分析師普遍上調目標價，惟地緣政治"
            "風險與美國關稅政策仍為主要不確定因素，需持續追蹤。"
        ),
        facts=[
            Fact(id=1, text="Q1 2026 營收 NT$839.3B，YoY +41.6%，優於市場預期",                  confidence=95, source="台積電法說會"),
            Fact(id=2, text="AI 加速器相關需求佔整體收入超過 20%，全年持續成長",                   confidence=90, source="法說會投影片"),
            Fact(id=3, text="CoWoS 先進封裝產能預計 2026 年底擴增至 2025 年底的 2.5 倍",           confidence=88, source="分析師報告"),
            Fact(id=4, text="2nm N2 製程已進入量產，良率達標，Apple / NVIDIA 等大客戶如期導入",    confidence=82, source="法說會 Q&A"),
            Fact(id=5, text="全年美元收入成長指引上調至「中段 20% 以上」",                         confidence=95, source="台積電法說會"),
        ],
        risks=[
            Risk(id=1, level="high",   text="美國對台半導體加徵關稅政策存在不確定性，可能影響客戶資本支出決策", source="路透社"),
            Risk(id=2, level="high",   text="地緣政治：台海情勢可能影響外資持股意願與全球供應鏈穩定性",        source="Bloomberg"),
            Risk(id=3, level="medium", text="PC / 手機端需求回溫速度低於預期，庫存調整週期恐延長",            source="集邦科技"),
            Risk(id=4, level="low",    text="Intel Foundry、三星技術追趕進度仍需持續觀察",                    source="IC Insights"),
        ],
        evidence=[
            Evidence(id=1, title="CEO 法說會原文聲明",
                content='"We are seeing very strong and broad-based demand from our AI customers. '
                        'CoWoS capacity is fully booked through the end of 2026, and we are '
                        'aggressively expanding capacity to support our customers." '
                        '— C.C. Wei, CEO, TSMC 2026Q1 Earnings Call',
                heat=95, tags=["AI需求", "CoWoS", "CEO直述"]),
            Evidence(id=2, title="財務數據對照",
                content="Q1 2026 vs Q1 2025｜營收：NT$839.3B vs NT$592.6B (+41.6%)｜"
                        "EPS：NT$23.74 vs NT$16.08 (+47.7%)｜毛利率：58.1% vs 52.7% (+5.4pp)｜"
                        "資本支出：NT$145B（在指引範圍內）",
                heat=88, tags=["財務數據", "YoY"]),
            Evidence(id=3, title="分析師評級共識",
                content="彭博統計 42 位追蹤分析師中，38 位給予「買入／增持」評級（90.5%），"
                        "3 位「持有」，1 位「賣出」。目標價中位數 NT$1,420，"
                        "較當前收盤有約 17% 上漲空間。",
                heat=74, tags=["分析師共識", "目標價"]),
        ],
        sources=[
            Source(id=1, title="台積電 2026Q1 法說會逐字稿",                                type="official", date="2026-04-17", relevance=98),
            Source(id=2, title="Bloomberg: TSMC Raises Revenue Outlook on AI Chip Boom",    type="media",    date="2026-04-18", relevance=85),
            Source(id=3, title="Morgan Stanley: 台積電目標價上調至 NT$1,450",               type="research", date="2026-04-19", relevance=78),
            Source(id=4, title="集邦科技：晶圓代工市場 2026 展望報告",                       type="research", date="2026-04-10", relevance=72),
        ],
    )

# ══════════════════════════════════════════════════════════════
#  API 端點
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "LLM理財 API 已啟動", "docs": "/docs"}


@app.post("/api/query", response_model=QueryResult)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查詢內容不得為空")
    time.sleep(0.3)
    result = build_mock_result(req.query)
    new_id = max((h.id for h in MOCK_HISTORY), default=0) + 1
    MOCK_HISTORY.insert(0, HistoryItem(
        id=new_id,
        query=req.query,
        time=datetime.now().strftime("%H:%M"),
        status=result.status,
    ))
    return result


@app.get("/api/history", response_model=list[HistoryItem])
def history(limit: int = 20):
    return MOCK_HISTORY[:limit]


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
