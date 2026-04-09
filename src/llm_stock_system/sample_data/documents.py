from datetime import datetime, timedelta, timezone

from llm_stock_system.core.enums import SourceTier, Topic
from llm_stock_system.core.models import Document


NOW = datetime.now(timezone.utc)


def days_ago(days: int, hours: int = 0) -> datetime:
    return NOW - timedelta(days=days, hours=hours)


SAMPLE_DOCUMENTS = [
    Document(
        ticker="2330",
        title="台積電公告第一季法說會時程",
        content="台積電公告第一季法說會時程，市場將關注 AI 需求、毛利率與資本支出更新。",
        source_name="台灣證交所",
        source_type="official_announcement",
        source_tier=SourceTier.HIGH,
        url="https://example.com/twse/2330/investor-conference",
        published_at=days_ago(2),
        topics=[Topic.ANNOUNCEMENT, Topic.NEWS],
    ),
    Document(
        ticker="2330",
        title="台積電最新一季財報聚焦毛利率與資本支出紀律",
        content="台積電最新一季財報顯示市場持續關注 AI 需求、毛利率表現與資本支出紀律。",
        source_name="公司投資人關係",
        source_type="earnings_release",
        source_tier=SourceTier.HIGH,
        url="https://example.com/ir/2330/q1-results",
        published_at=days_ago(3),
        topics=[Topic.EARNINGS, Topic.ANNOUNCEMENT],
    ),
    Document(
        ticker="2330",
        title="主流媒體追蹤台積電供應鏈與 AI 封裝需求",
        content="主流財經媒體整理台積電供應鏈需求與先進封裝產能動向，認為短期仍需觀察接單與資本支出節奏。",
        source_name="經濟日報",
        source_type="media",
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/media/2330/ai-demand",
        published_at=days_ago(4),
        topics=[Topic.NEWS],
    ),
    Document(
        ticker="2317",
        title="鴻海更新製造投資計畫",
        content="鴻海公告最新製造投資計畫，內容聚焦產能布局與資本投入安排。",
        source_name="公開資訊觀測站",
        source_type="official_announcement",
        source_tier=SourceTier.HIGH,
        url="https://example.com/mops/2317/investment-plan",
        published_at=days_ago(1, 6),
        topics=[Topic.ANNOUNCEMENT],
    ),
    Document(
        ticker="2344",
        title="華邦電近 30 天股價區間",
        content="根據近 30 天資料整理，華邦電最高價為 31.80 元，最低價為 24.60 元。",
        source_name="FinMind TaiwanStockPrice",
        source_type="market_data",
        source_tier=SourceTier.HIGH,
        url="https://example.com/price/2344/30d-range",
        published_at=days_ago(0, 4),
        topics=[Topic.NEWS],
    ),
    Document(
        ticker="2344",
        title="華邦電近 30 天收盤與波動摘要",
        content="華邦電近 30 天平均收盤價與區間波動可供觀察短線交易熱度，最高價為 31.80 元，最低價為 24.60 元。",
        source_name="FinMind TaiwanStockPrice",
        source_type="market_data_summary",
        source_tier=SourceTier.HIGH,
        url="https://example.com/price/2344/30d-summary",
        published_at=days_ago(0, 3),
        topics=[Topic.NEWS],
    ),
    Document(
        ticker="3680",
        title="ASML 展望轉保守，家登短線觀察 EUV 與光罩載具需求節奏",
        content="ASML 最新展望不如市場預期後，半導體設備鏈短線情緒轉趨保守。市場對家登的利空分析主要聚焦 EUV 與先進製程擴產節奏是否放慢，以及客戶資本支出是否延後。",
        source_name="經濟日報",
        source_type="media",
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/media/3680/asml-outlook",
        published_at=days_ago(1),
        topics=[Topic.NEWS],
    ),
    Document(
        ticker="3680",
        title="半導體設備族群氣氛偏觀望，家登留意先進製程資本支出變化",
        content="市場認為若國際設備大廠釋出較保守展望，設備族群短線評價容易承壓。對家登而言，觀察重點會回到晶圓廠資本支出、EUV 相關訂單能見度與先進製程擴產時程。",
        source_name="工商時報",
        source_type="media",
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/media/3680/capex-sentiment",
        published_at=days_ago(2),
        topics=[Topic.NEWS],
    ),
    Document(
        ticker="6187",
        title="ASML 財測不如預期，萬潤短線情緒轉趨觀望",
        content="ASML 財測偏保守後，市場對萬潤的解讀偏向短線情緒面利空，擔心設備訂單與客戶擴產節奏放緩，導致評價面先承受壓力。",
        source_name="MoneyDJ",
        source_type="media",
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/media/6187/asml-guidance",
        published_at=days_ago(1, 6),
        topics=[Topic.NEWS],
    ),
    Document(
        ticker="6187",
        title="設備股利空分析聚焦訂單能見度，萬潤觀察資本支出降溫風險",
        content="針對半導體設備族群的最新利空分析指出，若晶圓廠資本支出降溫，萬潤後續訂單能見度與市場情緒都可能受到影響，短線資金態度偏向保守。",
        source_name="鉅亨網",
        source_type="media",
        source_tier=SourceTier.MEDIUM,
        url="https://example.com/media/6187/order-visibility",
        published_at=days_ago(3),
        topics=[Topic.NEWS],
    ),
]
