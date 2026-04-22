"""
LLM理財 Dashboard — Streamlit 前端
================================
啟動：streamlit run frontend/app.py
後端 API：透過環境變數 BACKEND_URL 設定（預設 http://localhost:8000）
  - 本機開發：不設定，自動使用 localhost:8000
  - Streamlit Cloud：在 App settings > Secrets 填入 BACKEND_URL = "https://your-app.vercel.app"
"""

import os
import requests
import streamlit as st
from datetime import datetime

# ─────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────
# 讀取後端 URL：
#   - Streamlit Cloud：在 App settings > Secrets 加入 BACKEND_URL = "https://your-app.vercel.app"
#   - 本機開發：設環境變數 BACKEND_URL，或直接使用預設 localhost:8000
try:
    _BACKEND_URL = st.secrets.get("BACKEND_URL", os.environ.get("BACKEND_URL", "http://localhost:8000"))
except Exception:
    _BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_BACKEND_URL = (_BACKEND_URL or "http://localhost:8000").rstrip("/")
API_BASE = f"{_BACKEND_URL}/api"
TIMEOUT  = 30  # default; long-range queries use _timeout_for()

# Longer time ranges need more time to fetch & process data
_TIMEOUT_MAP = {
    "1d": 30, "7d": 30, "30d": 45,
    "latest_quarter": 60, "1y": 90, "3y": 120, "5y": 150,
}

def _timeout_for(time_val: str) -> int:
    return _TIMEOUT_MAP.get(time_val, TIMEOUT)

TOPIC_OPTIONS = {
    "綜合 (composite)": "composite",
    "新聞 (news)":      "news",
    "財報 (earnings)":  "earnings",
    "公告 (announcement)": "announcement",
}
TIME_OPTIONS = {
    "近 1 天":  "1d",
    "近 7 天":  "7d",
    "近 30 天": "30d",
    "最新一季": "latest_quarter",
    "近 1 年":  "1y",
    "近 3 年":  "3y",
    "近 5 年":  "5y",
}

# ─────────────────────────────────────────────────────────
#  Page config  ── 必須是第一個 Streamlit 呼叫
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM個股理財助理",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
#  Design Tokens
# ─────────────────────────────────────────────────────────
BG        = "#f5f5f7"
SURFACE   = "#ffffff"
CARD      = "#fefce8"
BORDER    = "#e2e4e9"
BORDER_HI = "#c8cbd4"
TEXT      = "#0d0d12"
SUB       = "#4a4a5e"
DIM       = "#9696aa"
ACCENT    = "#5b4fff"
GREEN     = "#16a34a"
GREEN_BG  = "#f0fdf4"
YELLOW    = "#b45309"
YELLOW_BG = "#fffbeb"
RED       = "#dc2626"
RED_BG    = "#fef2f2"
BLUE      = "#1d6fe8"
PURPLE    = "#7c3aed"

# ─────────────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── Light background everywhere ── */
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  section[data-testid="stSidebarContent"],
  [data-testid="stHeader"] {{
    background-color: {BG} !important;
  }}

  /* ── Block container: full width, tight padding ── */
  .block-container {{
    padding: 0.6rem 1.2rem 3rem !important;
    max-width: 100% !important;
  }}

  /* ── Column layout: no extra gap ── */
  [data-testid="stHorizontalBlock"] {{ gap: 0 !important; }}

  /* ── Typography ── */
  p, span, li, label, div {{ color: {SUB}; font-family: -apple-system, "Noto Sans TC", sans-serif; }}
  h1, h2, h3, h4           {{ color: {TEXT}; }}

  /* ── Text input ── */
  .stTextInput > div > div {{
    background: {SURFACE} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 9px !important;
  }}
  .stTextInput input {{
    color: {TEXT} !important;
    font-size: 13px !important;
    background: transparent !important;
  }}
  .stTextInput input::placeholder {{ color: {DIM} !important; }}

  /* ── Select boxes ── */
  [data-testid="stSelectbox"] > div > div {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 7px !important;
  }}
  [data-testid="stSelectbox"] span {{ color: {TEXT} !important; font-size: 12px !important; }}

  /* ── Primary button (查詢) ── */
  .stButton > button[kind="primary"],
  .stButton > button:not([kind="secondary"]) {{
    background: {ACCENT} !important;
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.4) !important;
    border: none !important;
    font-weight: 800 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    padding: 5px 18px !important;
  }}
  .stButton > button[kind="primary"]:hover {{ opacity: 0.85 !important; }}

  /* ── Secondary buttons (history items) ── */
  .stButton > button[kind="secondary"] {{
    background: transparent !important;
    color: {SUB} !important;
    border: 1px solid {BORDER} !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    padding: 4px 10px !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.5 !important;
  }}
  .stButton > button[kind="secondary"]:hover {{
    background: {SURFACE} !important;
    border-color: {BORDER_HI} !important;
    color: {TEXT} !important;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    gap: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {DIM} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 18px !important;
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom: 2px solid {ACCENT} !important;
    background: transparent !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
    background: {CARD};
    padding: 10px 12px;
    border: 1px solid {BORDER};
    border-top: none;
    border-radius: 0 0 8px 8px;
  }}

  /* ── Expander ── */
  .stExpander {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    background: {CARD} !important;
    margin-bottom: 6px !important;
  }}
  .stExpander details summary p {{
    color: {TEXT} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }}
  .stExpander details {{ background: {CARD} !important; }}
  .stExpander [data-testid="stExpanderDetails"] {{ padding: 8px 12px !important; }}

  /* ── Divider ── */
  hr {{ border-color: {BORDER} !important; margin: 0.5rem 0 !important; }}

  /* ── Spinner ── */
  .stSpinner div {{ border-top-color: {ACCENT} !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {BORDER_HI}; }}

  /* ── Column right border (separator) ── */
  div[data-testid="stVerticalBlockBorderWrapper"]:first-of-type {{
    border-right: 1px solid {BORDER};
    padding-right: 10px;
  }}

  /* ── Card shadow for light mode ── */
  .stExpander,
  [data-testid="stVerticalBlock"] > div > div {{
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────────────────
_defaults = {
    "result":              None,
    "history":             [],
    "last_query_text":     "",
    "chart_ticker":        None,
    "api_error":           None,
    "show_conflict":       False,
    "conflict_detail":     None,   # cached result from /api/sources/{query_id}
    "conflict_query_id":   None,   # which query_id the cached detail belongs to
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────
#  API Helpers
# ─────────────────────────────────────────────────────────
def api_query(query: str, ticker: str | None, topic_val: str, time_val: str) -> dict:
    payload: dict = {"query": query}
    if ticker and ticker.strip():
        payload["ticker"] = ticker.strip()
    if topic_val and topic_val != "composite":
        payload["topic"] = topic_val
    if time_val:
        payload["timeRange"] = time_val
    resp = requests.post(f"{API_BASE}/query", json=payload, timeout=_timeout_for(time_val))
    resp.raise_for_status()
    return resp.json()

def api_health() -> dict | None:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return None

def api_sources(query_id: str) -> dict | None:
    """Fetch detailed source breakdown from /api/sources/{query_id}."""
    try:
        resp = requests.get(f"{API_BASE}/sources/{query_id}", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────
#  HTML / CSS micro-helpers
# ─────────────────────────────────────────────────────────
def _g(d: dict, *keys, default=None):
    """Get first existing key, supporting camelCase & snake_case aliases."""
    for k in keys:
        if k in d:
            return d[k]
    return default

def _conf_color(light: str) -> str:
    return {"green": GREEN, "yellow": YELLOW, "red": RED}.get((light or "").lower(), YELLOW)

def _conf_label(light: str) -> str:
    return {"green": "高信心", "yellow": "中信心", "red": "低信心"}.get((light or "").lower(), "—")

def _pill(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;'
        f'background:{color}18;border:1px solid {color}28;border-radius:20px;'
        f'font-size:10px;font-weight:700;color:{color}">{text}</span>'
    )

def _bar(v: int, color: str) -> str:
    return (
        f'<div style="display:inline-flex;align-items:center;gap:4px">'
        f'<div style="width:32px;height:3px;background:{BORDER};border-radius:2px;overflow:hidden">'
        f'<div style="width:{v}%;height:100%;background:{color}"></div></div>'
        f'<span style="font-size:10px;color:{DIM}">{v}%</span></div>'
    )

def _sec(icon: str, label: str, sub: str = "") -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">'
        f'<span>{icon}</span>'
        f'<span style="font-size:11px;font-weight:700;color:{SUB};letter-spacing:.3px">{label}</span>'
        + (f'<span style="font-size:10px;color:{DIM}">{sub}</span>' if sub else "")
        + "</div>"
    )

def _card(inner: str, border: str = BORDER, bg: str = CARD, pad: str = "14px 16px") -> str:
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        f'padding:{pad};margin-bottom:10px">{inner}</div>'
    )

def _row_item(dot_color: str, text: str) -> str:
    return (
        f'<div style="font-size:13.5px;color:{TEXT};line-height:1.75;padding:8px 10px;'
        f'border-radius:6px;background:{CARD};border:1px solid {BORDER};margin-bottom:4px">'
        f'<span style="color:{dot_color};margin-right:6px;font-weight:700">·</span>{text}</div>'
    )

# ─────────────────────────────────────────────────────────
#  NAV BAR
# ─────────────────────────────────────────────────────────
logo_col, search_col, ticker_col, topic_col, time_col, btn_col = st.columns(
    [2.2, 3.5, 1, 1.3, 1.1, 0.7]
)

with logo_col:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding-top:4px">'
        f'<span style="font-size:26px;font-weight:800;color:{TEXT};letter-spacing:-0.5px;white-space:nowrap">'
        f'LLM個股理財助理</span>'
        f'<span style="font-size:9px;color:{DIM};background:{SURFACE};padding:1px 5px;'
        f'border-radius:3px;border:1px solid {BORDER};flex-shrink:0">Beta</span></div>',
        unsafe_allow_html=True,
    )

with search_col:
    query_input = st.text_input(
        "query",
        placeholder="輸入查詢，例如：台積電法說會重點、AI 需求展望、近期風險評估…",
        label_visibility="collapsed",
        key="query_input",
    )

with ticker_col:
    ticker_input = st.text_input(
        "ticker",
        placeholder="股票代號（選填）",
        label_visibility="collapsed",
        key="ticker_input",
    )

with topic_col:
    topic_label = st.selectbox(
        "主題",
        list(TOPIC_OPTIONS.keys()),
        label_visibility="collapsed",
        key="topic_sel",
    )

with time_col:
    time_label = st.selectbox(
        "時間",
        list(TIME_OPTIONS.keys()),
        index=1,  # 預設「近 7 天」
        label_visibility="collapsed",
        key="time_sel",
    )

with btn_col:
    do_query = st.button("查詢", use_container_width=True, type="primary")

st.markdown(f'<hr style="margin:.3rem 0 .6rem">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  Handle Query Submission
# ─────────────────────────────────────────────────────────
if do_query and query_input.strip():
    st.session_state.api_error       = None
    st.session_state.last_query_text = query_input
    st.session_state.show_conflict   = False
    st.session_state.conflict_detail = None
    st.session_state.conflict_query_id = None
    if ticker_input and ticker_input.strip():
        st.session_state.chart_ticker = ticker_input.strip()

    _time_val = TIME_OPTIONS[time_label]
    _spinner_msg = (
        "資料範圍較大，分析需要較長時間，請耐心等候（最多約 %d 秒）…" % _timeout_for(_time_val)
        if _timeout_for(_time_val) > 30
        else "分析中，請稍候…"
    )
    with st.spinner(_spinner_msg):
        try:
            result = api_query(
                query    = query_input,
                ticker   = ticker_input or None,
                topic_val= TOPIC_OPTIONS[topic_label],
                time_val = _time_val,
            )
            st.session_state.result = result

            light = _g(result, "confidenceLight", "confidence_light", default="yellow")
            st.session_state.history.insert(0, {
                "query":  query_input,
                "ticker": ticker_input or "",
                "time":   datetime.now().strftime("%H:%M"),
                "light":  light,
                "result": result,
                "time_range": _time_val,
            })
            if len(st.session_state.history) > 25:
                st.session_state.history = st.session_state.history[:25]

        except requests.exceptions.ConnectionError:
            st.session_state.api_error = "connect"
        except requests.exceptions.HTTPError as e:
            st.session_state.api_error = f"http:{e.response.status_code}"
        except Exception as e:
            st.session_state.api_error = str(e)[:200]

# ─────────────────────────────────────────────────────────
#  STOCK PRICE CHART (spans center + right, above results)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price(ticker: str, period: str) -> "tuple[object, str] | tuple[None, None]":
    """Fetch OHLCV from yfinance; tries .TW then .TWO suffixes."""
    try:
        import yfinance as yf
        for suffix in (".TW", ".TWO"):
            sym = f"{ticker}{suffix}"
            hist = yf.Ticker(sym).history(period=period)
            if not hist.empty:
                return hist, sym
        return None, None
    except Exception:
        return None, None

_PERIOD_MAP = {
    "1d": "5d",   # yfinance '1d' = only today's intraday; use 5d for a visible line
    "7d": "7d",
    "30d": "1mo",
    "latest_quarter": "3mo",
    "1y": "1y",
    "3y": "3y",
    "5y": "5y",
}

_chart_ticker = (
    ticker_input.strip() if (ticker_input and ticker_input.strip())
    else st.session_state.chart_ticker
)
if _chart_ticker:
    _, chart_col = st.columns([1.4, 6.4])
    with chart_col:
        _cur_time_val = TIME_OPTIONS.get(time_label, "7d")
        _period = _PERIOD_MAP.get(_cur_time_val, "1mo")
        _hist, _sym = _fetch_price(_chart_ticker, _period)

        if _hist is not None:
            import plotly.graph_objects as go

            # Build a clean area / candlestick chart
            _has_ohlc = all(c in _hist.columns for c in ("Open", "High", "Low", "Close"))
            fig = go.Figure()

            if _has_ohlc and _cur_time_val in ("1d", "7d", "30d", "latest_quarter"):
                # Candlestick for shorter ranges
                fig.add_trace(go.Candlestick(
                    x=_hist.index,
                    open=_hist["Open"],
                    high=_hist["High"],
                    low=_hist["Low"],
                    close=_hist["Close"],
                    increasing_line_color=GREEN,
                    decreasing_line_color=RED,
                    increasing_fillcolor="rgba(62,207,142,0.55)",
                    decreasing_fillcolor="rgba(240,96,96,0.55)",
                    name=_chart_ticker,
                    showlegend=False,
                ))
            else:
                # Area line for longer ranges
                fig.add_trace(go.Scatter(
                    x=_hist.index,
                    y=_hist["Close"],
                    mode="lines",
                    line=dict(color=ACCENT, width=1.8),
                    fill="tozeroy",
                    fillcolor="rgba(124,111,255,0.09)",
                    name=_chart_ticker,
                    showlegend=False,
                ))

            fig.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=4, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickfont=dict(color=DIM, size=10),
                    rangeslider=dict(visible=False),
                    showline=False,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=BORDER,
                    zeroline=False,
                    tickfont=dict(color=DIM, size=10),
                    side="right",
                    showline=False,
                ),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor=CARD,
                    bordercolor=BORDER_HI,
                    font=dict(color=TEXT, size=11),
                ),
            )

            # Ticker label above chart
            _close_last = _hist["Close"].iloc[-1] if not _hist.empty else None
            _close_prev = _hist["Close"].iloc[-2] if len(_hist) > 1 else _close_last
            _chg = ((_close_last - _close_prev) / _close_prev * 100) if (_close_last and _close_prev) else 0
            _chg_color = GREEN if _chg >= 0 else RED
            _chg_sign  = "+" if _chg >= 0 else ""
            st.markdown(
                f'<div style="font-size:11px;color:{SUB};font-weight:600;margin-bottom:2px">'
                f'<span style="color:{TEXT}">{_sym}</span>'
                f' &nbsp;<span style="color:{TEXT};font-size:13px;font-weight:700">'
                f'{_close_last:.2f}</span>'
                f' &nbsp;<span style="color:{_chg_color};font-size:11px">'
                f'{_chg_sign}{_chg:.2f}%</span>'
                f'&nbsp;<span style="color:{DIM};font-size:10px">({_period})</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                f'<div style="font-size:11px;color:{DIM};padding:6px 0">'
                f'無法取得 {_chart_ticker} 的股價資料（僅支援台股代號）</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────
#  THREE-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────
col_hist, col_main, col_src = st.columns([1.4, 4.1, 2.3])

# ══════════════════════════════════════════
#  COL A — History + System Status
# ══════════════════════════════════════════
with col_hist:
    st.markdown(
        f'<div style="font-size:9px;color:{DIM};text-transform:uppercase;'
        f'letter-spacing:.7px;font-weight:700;margin-bottom:8px;padding-right:8px">查詢歷史</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.markdown(
            f'<div style="font-size:11px;color:{DIM};padding:8px 0">尚無查詢記錄</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, item in enumerate(st.session_state.history):
            dot_col = _conf_color(item["light"])
            label_str = (
                f'{"●" if item["light"]=="green" else ("◐" if item["light"]=="yellow" else "○")} '
                f'{item["query"][:40]}{"…" if len(item["query"])>40 else ""}'
                f'\n{item["time"]}'
                + (f' · {item["ticker"]}' if item.get("ticker") else "")
            )
            if st.button(label_str, key=f"hist_{i}", type="secondary", use_container_width=True):
                st.session_state.result = item["result"]
                st.session_state.last_query_text = item["query"]
                st.rerun()

    # System health
    st.markdown(
        f'<div style="margin-top:14px;border-top:1px solid {BORDER};padding-top:10px"></div>',
        unsafe_allow_html=True,
    )
    health = api_health()
    if health:
        mode    = health.get("mode", "—")
        llm_m   = health.get("llmMode", "—")
        news_ok = health.get("newsPipelineEnabled", False)
        embed   = health.get("embeddingEnabled", False)
        st.markdown(
            f'<div style="font-size:10px;color:{DIM};line-height:1.8;padding-right:6px">'
            f'<div style="color:{GREEN};font-weight:700;margin-bottom:4px">● 後端連線正常</div>'
            f'<div>模式：<span style="color:{SUB}">{mode}</span></div>'
            f'<div>LLM：<span style="color:{SUB}">{llm_m}</span></div>'
            f'<div>新聞管線：<span style="color:{"{"}{GREEN}{"}" if news_ok else RED}">{"啟用" if news_ok else "停用"}</span></div>'
            f'<div>Embedding：<span style="color:{"{"}{GREEN}{"}" if embed else DIM}">{"啟用" if embed else "停用"}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:10px;color:{RED};padding-right:6px">'
            f'● 後端離線<br>'
            f'<span style="color:{DIM};font-size:9px">請執行：<br>uvicorn llm_stock_system.api.app:create_app --factory</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════
#  COL B — Main Results
# ══════════════════════════════════════════
with col_main:

    # ── Error state ────────────────────────
    if st.session_state.api_error:
        err = st.session_state.api_error
        if err == "connect":
            msg = "無法連線至後端 API (localhost:8000)。請確認後端已啟動。"
        elif err.startswith("http:"):
            msg = f"API 回傳錯誤碼 {err[5:]}，請檢查後端日誌。"
        else:
            msg = f"發生例外：{err}"
        st.markdown(_card(
            f'<div style="text-align:center;padding:24px 0">'
            f'<div style="font-size:28px;margin-bottom:10px">⚠️</div>'
            f'<div style="font-size:14px;color:{RED};font-weight:700;margin-bottom:8px">查詢失敗</div>'
            f'<div style="font-size:12px;color:{SUB};line-height:1.8">{msg}</div>'
            f'</div>',
            border=f"{RED}40",
        ), unsafe_allow_html=True)

    # ── Empty state ─────────────────────────
    elif st.session_state.result is None:
        st.markdown(_card(
            f'<div style="text-align:center;padding:48px 0">'
            f'<div style="font-size:38px;margin-bottom:14px">🔍</div>'
            f'<div style="font-size:15px;color:{SUB};font-weight:500;margin-bottom:6px">輸入查詢開始分析</div>'
            f'<div style="font-size:12px;color:{DIM};line-height:1.9">'
            f'法說會重點 · 財報分析 · 產業展望 · 風險評估<br>'
            f'支援指定股票代號與時間範圍</div>'
            f'</div>'
        ), unsafe_allow_html=True)

    # ── Result state ────────────────────────
    else:
        r = st.session_state.result

        # Parse fields (handle both camelCase and snake_case)
        light       = _g(r, "confidenceLight", "confidence_light", default="yellow")
        score       = _g(r, "confidenceScore",  "confidence_score",  default=0.0)
        summary     = r.get("summary", "")
        highlights  = r.get("highlights", [])
        facts       = r.get("facts", [])
        impacts     = r.get("impacts", [])
        risks       = r.get("risks", [])
        warnings    = r.get("warnings", [])
        disclaimer  = r.get("disclaimer", "")
        q_id        = r.get("query_id", "")
        q_profile   = _g(r, "queryProfile", "query_profile", default="legacy")
        clf_source  = _g(r, "classifierSource", "classifier_source", default="rule")

        col_light   = _conf_color(light)
        conf_pct    = int((score or 0) * 100)
        conf_lbl    = _conf_label(light)

        # ── Red (low confidence) state ──────
        if light == "red":
            st.markdown(_card(
                f'<div style="text-align:center;padding:24px 0">'
                f'<div style="font-size:28px;margin-bottom:10px">⚠️</div>'
                f'<div style="font-size:14px;color:{RED};font-weight:700;margin-bottom:10px">'
                f'資料不足，無法提供可靠回應</div>'
                f'<div style="font-size:12px;color:{SUB};line-height:1.85;'
                f'max-width:480px;margin:0 auto">{summary}</div>'
                f'</div>',
                border=f"{RED}35",
                bg=RED_BG,
            ), unsafe_allow_html=True)

        # ── Green / Yellow state ─────────────
        else:
            # Confidence header
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">'
                + _pill(f"🛡 {conf_lbl} {conf_pct}%", col_light)
                + '</div>',
                unsafe_allow_html=True,
            )

            # Summary card
            st.markdown(_card(
                _sec("📋", "摘要")
                + f'<p style="font-size:14.5px;line-height:1.9;color:{TEXT};margin:0">{summary}</p>'
            ), unsafe_allow_html=True)

            # Highlights
            if highlights:
                items_html = "".join(
                    f'<div style="font-size:13.5px;color:{TEXT};padding:8px 10px;'
                    f'background:{CARD};border-radius:6px;border:1px solid {BORDER};'
                    f'margin-bottom:4px;line-height:1.7">'
                    f'<span style="color:{ACCENT};margin-right:6px;font-weight:700">▸</span>{h}</div>'
                    for h in highlights
                )
                st.markdown(_card(
                    _sec("✨", "重要重點") + items_html, pad="12px 14px"
                ), unsafe_allow_html=True)

            # Facts / Impacts / Risks — tabs
            tab_f, tab_i, tab_r = st.tabs([
                f"✅ 已核實事實 ({len(facts)})",
                f"💡 可能影響 ({len(impacts)})",
                f"⚠️ 風險提示 ({len(risks)})",
            ])

            with tab_f:
                if facts:
                    st.markdown(
                        "".join(_row_item(GREEN, item) for item in facts),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="font-size:12px;color:{DIM};padding:10px 0">暫無事實資料</div>',
                        unsafe_allow_html=True,
                    )

            with tab_i:
                if impacts:
                    st.markdown(
                        "".join(_row_item(BLUE, item) for item in impacts),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="font-size:12px;color:{DIM};padding:10px 0">暫無影響資料</div>',
                        unsafe_allow_html=True,
                    )

            with tab_r:
                if risks:
                    st.markdown(
                        "".join(_row_item(YELLOW, item) for item in risks),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="font-size:12px;color:{DIM};padding:10px 0">暫無風險資料</div>',
                        unsafe_allow_html=True,
                    )

        # Warnings — filter out developer-only technical messages
        _TECHNICAL_PREFIXES = (
            "Preferred facets not synced",
            "Required facets not synced",
            "facet",
            "Digest 品質 gate",
            "Answer indicates",
        )
        user_warnings = [
            w for w in warnings
            if not any(w.startswith(p) for p in _TECHNICAL_PREFIXES)
        ]
        if user_warnings:
            st.markdown(
                "".join(
                    f'<div style="font-size:11px;color:{YELLOW};padding:5px 10px;'
                    f'background:{YELLOW_BG};border-radius:6px;border:1px solid {YELLOW}28;'
                    f'margin-top:6px">⚠ {w}</div>'
                    for w in user_warnings
                ),
                unsafe_allow_html=True,
            )

        # Disclaimer
        if disclaimer:
            st.markdown(
                f'<div style="font-size:10px;color:{DIM};margin-top:10px;padding:8px 10px;'
                f'border-radius:6px;border:1px solid {BORDER};background:{SURFACE};'
                f'line-height:1.65">{disclaimer}</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════
#  COL C — Sources + Data Status
# ══════════════════════════════════════════
with col_src:
    st.markdown(
        f'<div style="font-size:9px;color:{DIM};text-transform:uppercase;'
        f'letter-spacing:.7px;font-weight:700;margin-bottom:8px;padding-left:8px">來源 / 資料品質</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.result:
        st.markdown(
            f'<div style="font-size:11px;color:{DIM};padding:20px 8px;text-align:center">'
            f'查詢後顯示來源與資料狀態</div>',
            unsafe_allow_html=True,
        )
    else:
        r       = st.session_state.result
        sources = r.get("sources", [])
        ds      = _g(r, "dataStatus", "data_status", default={}) or {}

        # ── Data quality card ───────────────
        if ds:
            suf = ds.get("sufficiency", "")
            con = ds.get("consistency", "")
            fre = ds.get("freshness",   "")

            suf_col = GREEN if suf == "sufficient"        else RED
            con_col = (GREEN  if con == "consistent"
                       else YELLOW if con == "mostly_consistent"
                       else RED)
            fre_col = (GREEN  if fre == "recent"
                       else YELLOW if fre == "stale"
                       else RED)

            label_map = {
                "sufficient":        "充足",
                "insufficient":      "不足",
                "consistent":        "一致",
                "mostly_consistent": "大致一致",
                "conflicting":       "有矛盾",
                "recent":            "新鮮",
                "stale":             "稍舊",
                "outdated":          "過時",
            }

            def _status_row(label: str, val: str, col: str, clickable: bool = False) -> str:
                suffix = (' <span style="font-size:9px;opacity:.6;cursor:pointer">🔍</span>'
                          if clickable else "")
                return (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;font-size:11px;margin-bottom:5px">'
                    f'<span style="color:{DIM}">{label}</span>'
                    f'<span style="color:{col};font-weight:700">{label_map.get(val, val or "—")}{suffix}</span></div>'
                )

            is_conflicting = (con == "conflicting")

            st.markdown(
                _card(
                    _sec("📊", "資料品質")
                    + _status_row("充足性", suf, suf_col)
                    + _status_row("一致性", con, con_col, clickable=is_conflicting)
                    + _status_row("新鮮度", fre, fre_col),
                    pad="12px 14px",
                ),
                unsafe_allow_html=True,
            )

            # ── Conflict analysis section ────
            if is_conflicting:
                q_id = r.get("query_id", "")
                if st.button(
                    "🔍 查看矛盾詳情",
                    key="btn_conflict",
                    type="secondary",
                    use_container_width=True,
                ):
                    # Toggle visibility
                    st.session_state.show_conflict = not st.session_state.show_conflict
                    # Fetch detailed sources if we haven't for this query yet
                    if (q_id and
                            st.session_state.conflict_query_id != q_id and
                            st.session_state.show_conflict):
                        st.session_state.conflict_detail = api_sources(q_id)
                        st.session_state.conflict_query_id = q_id

                if st.session_state.show_conflict:
                    # Use sources already in the response (always available)
                    # Augment with any extra data from /api/sources if we have it
                    src_detail = st.session_state.conflict_detail or {}
                    detail_sources = src_detail.get("sources", []) if src_detail else []

                    # Build conflict view from response sources
                    all_src = list(sources)  # already fetched above
                    if not all_src and detail_sources:
                        all_src = detail_sources

                    # Sort by support_score descending
                    all_src_sorted = sorted(
                        all_src,
                        key=lambda s: s.get("support_score", 0) or 0,
                        reverse=True,
                    )

                    # Split: top half = consensus view, bottom half = dissenting
                    mid = max(1, len(all_src_sorted) // 2)
                    consensus_group  = all_src_sorted[:mid]
                    dissenting_group = all_src_sorted[mid:]

                    st.markdown(
                        f'<div style="border:1px solid {RED}40;border-radius:10px;'
                        f'background:{RED_BG};padding:12px 14px;margin-bottom:10px">'
                        f'<div style="font-size:11px;font-weight:700;color:{RED};margin-bottom:8px">'
                        f'⚡ 矛盾分析</div>'
                        f'<div style="font-size:10px;color:{SUB};line-height:1.6;margin-bottom:8px">'
                        f'系統偵測到以下來源對同一主題存在分歧。'
                        f'「一致性」分數是由 DataGovernanceLayer 比對各來源摘錄後計算得出，'
                        f'當多篇文件在關鍵數字或結論上相互矛盾時，會標記為「有矛盾」。'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                    def _conflict_src_card(s: dict, accent_col: str, badge: str) -> str:
                        t      = s.get("title", "未知來源")
                        sname  = s.get("source_name", "")
                        exc    = s.get("excerpt", "（無摘要）")
                        sc     = int((s.get("support_score", 0) or 0) * 100)
                        pub    = (s.get("published_at", "") or "")[:10]
                        return (
                            f'<div style="background:{SURFACE};border:1px solid {accent_col}30;'
                            f'border-radius:8px;padding:10px 12px;margin-bottom:6px">'
                            f'<div style="display:flex;justify-content:space-between;'
                            f'align-items:center;margin-bottom:5px">'
                            f'<span style="font-size:10px;font-weight:700;color:{accent_col}">'
                            f'{badge}</span>'
                            + _bar(sc, accent_col)
                            + f'</div>'
                            f'<div style="font-size:10px;color:{DIM};margin-bottom:4px">'
                            f'{t}' + (f' — {sname}' if sname else '') + (f' ({pub})' if pub else '')
                            + f'</div>'
                            f'<div style="font-size:11px;color:{SUB};line-height:1.65;'
                            f'font-style:italic;padding:6px 8px;background:{CARD};'
                            f'border-radius:5px;border:1px solid {BORDER}">{exc}</div>'
                            f'</div>'
                        )

                    if consensus_group:
                        st.markdown(
                            f'<div style="font-size:10px;color:{GREEN};font-weight:700;'
                            f'letter-spacing:.3px;margin:6px 0 4px;padding-left:2px">'
                            f'● 主流觀點（高支持度）</div>'
                            + "".join(
                                _conflict_src_card(s, GREEN, f"支持度 {int((s.get('support_score',0) or 0)*100)}%")
                                for s in consensus_group
                            ),
                            unsafe_allow_html=True,
                        )

                    if dissenting_group:
                        st.markdown(
                            f'<div style="font-size:10px;color:{RED};font-weight:700;'
                            f'letter-spacing:.3px;margin:6px 0 4px;padding-left:2px">'
                            f'○ 有異議觀點（低支持度）</div>'
                            + "".join(
                                _conflict_src_card(s, YELLOW, f"支持度 {int((s.get('support_score',0) or 0)*100)}%")
                                for s in dissenting_group
                            ),
                            unsafe_allow_html=True,
                        )

                    if not all_src:
                        st.markdown(
                            f'<div style="font-size:11px;color:{DIM};padding:8px">來源資料不足，無法顯示矛盾細節。</div>',
                            unsafe_allow_html=True,
                        )

        # ── Sources ─────────────────────────
        if sources:
            st.markdown(
                _sec("📄", "資料來源", f"共 {len(sources)} 筆"),
                unsafe_allow_html=True,
            )

            # Compute median support score to flag outlier sources
            support_scores = [
                s.get("support_score", 0) or 0 for s in sources
            ]
            avg_support = (sum(support_scores) / len(support_scores)) if support_scores else 0.5
            conflict_active = (ds.get("consistency", "") == "conflicting")

            for src in sources:
                title    = src.get("title",       "未知來源")
                src_name = src.get("source_name", "")
                tier     = src.get("source_tier", "medium")
                url      = src.get("url",         "#") or "#"
                pub_raw  = src.get("published_at", "")
                pub      = pub_raw[:10] if pub_raw else ""
                excerpt  = src.get("excerpt", "")
                supp_raw = src.get("support_score", 0)
                supp     = int((supp_raw or 0) * 100)
                corr     = src.get("corroboration_count", 1)

                tier_col   = {"high": GREEN, "medium": BLUE, "low": YELLOW}.get(tier, BLUE)
                tier_label = {"high": "高可信", "medium": "一般", "low": "較低"}.get(tier, tier)

                # Flag this source as a dissenting outlier if consistency is conflicting
                is_dissent = (conflict_active and (supp_raw or 0) < avg_support - 0.15)
                expander_label = ("⚡ " if is_dissent else "") + title

                with st.expander(expander_label, expanded=False):
                    meta = f"{src_name}" + (f" · {pub}" if pub else "")
                    conflict_badge = (
                        _pill("⚡ 有異議", RED) if is_dissent else
                        _pill("✓ 主流", GREEN) if conflict_active else ""
                    )
                    st.markdown(
                        f'<div style="font-size:10px;color:{DIM};margin-bottom:7px">{meta}</div>'
                        + f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:9px;flex-wrap:wrap">'
                        + _pill(tier_label, tier_col)
                        + _bar(supp, tier_col)
                        + f'<span style="font-size:10px;color:{DIM}">×{corr} 篇驗證</span>'
                        + conflict_badge
                        + '</div>'
                        + (
                            f'<div style="font-size:11px;color:{SUB};line-height:1.75;'
                            f'font-style:italic;padding:8px 10px;background:{SURFACE};'
                            f'border-radius:6px;border:1px solid {"" + RED + "30" if is_dissent else BORDER}">{excerpt}</div>'
                            if excerpt else ""
                        )
                        + (
                            f'<div style="margin-top:8px">'
                            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                            f'style="font-size:10px;color:{ACCENT};text-decoration:none">'
                            f'查看原文 ↗</a></div>'
                            if url != "#" else ""
                        ),
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f'<div style="font-size:11px;color:{DIM};padding:16px 8px;text-align:center">暫無來源資料</div>',
                unsafe_allow_html=True,
            )
