"""
Indonesian Market Dashboard — Deep Analysis Edition
5-layer confidence scoring: Macro → Sector → Catalysts → Fundamentals → Technical
Run with: python -m streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import feedparser
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Indo Deep Analysis", page_icon="🇮🇩", layout="wide")

# ── Stock Universe ─────────────────────────────────────────────────────────────

STOCKS = {
    # Banking
    "BCA (BBCA)":         {"ticker": "BBCA.JK", "sector": "Banking",   "commodity": None,
                           "search": "BCA Bank Central Asia"},
    "BRI (BBRI)":         {"ticker": "BBRI.JK", "sector": "Banking",   "commodity": None,
                           "search": "BRI Bank Rakyat Indonesia"},
    "Mandiri (BMRI)":     {"ticker": "BMRI.JK", "sector": "Banking",   "commodity": None,
                           "search": "Bank Mandiri Indonesia"},
    "BNI (BBNI)":         {"ticker": "BBNI.JK", "sector": "Banking",   "commodity": None,
                           "search": "BNI Bank Negara Indonesia"},
    "BTN (BBTN)":         {"ticker": "BBTN.JK", "sector": "Banking",   "commodity": None,
                           "search": "BTN Bank Tabungan Negara"},
    # Coal
    "Bayan Coal (BYAN)":  {"ticker": "BYAN.JK", "sector": "Coal",      "commodity": "MTF=F",
                           "search": "Bayan Resources coal Indonesia"},
    "Adaro (ADRO)":       {"ticker": "ADRO.JK", "sector": "Coal",      "commodity": "MTF=F",
                           "search": "Adaro Energy coal Indonesia"},
    "Bukit Asam (PTBA)":  {"ticker": "PTBA.JK", "sector": "Coal",      "commodity": "MTF=F",
                           "search": "Bukit Asam coal Indonesia"},
    "Harum Energy (HRUM)":{"ticker": "HRUM.JK", "sector": "Coal",      "commodity": "MTF=F",
                           "search": "Harum Energy coal Indonesia"},
    # Mining
    "Vale Indonesia (INCO)":  {"ticker": "INCO.JK",  "sector": "Mining", "commodity": "JJN",
                               "search": "Vale Indonesia nickel"},
    "Aneka Tambang (ANTM)":   {"ticker": "ANTM.JK",  "sector": "Mining", "commodity": "GC=F",
                               "search": "Antam Aneka Tambang gold nickel"},
    "Merdeka Copper (MDKA)":  {"ticker": "MDKA.JK",  "sector": "Mining", "commodity": "CPER",
                               "search": "Merdeka Copper Gold Indonesia"},
    # Consumer
    "Indofood CB (ICBP)": {"ticker": "ICBP.JK", "sector": "Consumer",  "commodity": None,
                           "search": "Indofood CBP consumer Indonesia"},
    "Indofood (INDF)":    {"ticker": "INDF.JK", "sector": "Consumer",  "commodity": None,
                           "search": "Indofood consumer goods Indonesia"},
    "Unilever (UNVR)":    {"ticker": "UNVR.JK", "sector": "Consumer",  "commodity": None,
                           "search": "Unilever Indonesia consumer"},
    "Kalbe Farma (KLBF)": {"ticker": "KLBF.JK", "sector": "Healthcare","commodity": None,
                           "search": "Kalbe Farma pharmaceutical Indonesia"},
    # Telco
    "Telkom (TLKM)":      {"ticker": "TLKM.JK", "sector": "Telco",     "commodity": None,
                           "search": "Telkom Indonesia telecommunications"},
    "Indosat (ISAT)":     {"ticker": "ISAT.JK", "sector": "Telco",     "commodity": None,
                           "search": "Indosat Ooredoo telecommunications Indonesia"},
    "XL Axiata (EXCL)":   {"ticker": "EXCL.JK", "sector": "Telco",     "commodity": None,
                           "search": "XL Axiata telecommunications Indonesia"},
    # Energy
    "Medco Energy (MEDC)":{"ticker": "MEDC.JK", "sector": "Energy",    "commodity": "CL=F",
                           "search": "Medco Energy oil gas Indonesia"},
    # Property
    "Ciputra (CTRA)":     {"ticker": "CTRA.JK", "sector": "Property",  "commodity": None,
                           "search": "Ciputra property developer Indonesia"},
    "Bumi Serpong (BSDE)":{"ticker": "BSDE.JK", "sector": "Property",  "commodity": None,
                           "search": "Bumi Serpong Damai property Indonesia"},
    # Conglomerate
    "Astra (ASII)":       {"ticker": "ASII.JK", "sector": "Conglomerate","commodity": None,
                           "search": "Astra International Indonesia"},
    "GOTO":               {"ticker": "GOTO.JK", "sector": "Technology", "commodity": None,
                           "search": "GOTO Gojek Tokopedia Indonesia technology"},
}

# Sector descriptions and what drives them
SECTOR_CONTEXT = {
    "Banking": {
        "drivers": ["BI interest rate", "Rupiah stability", "loan growth", "NPL ratio", "foreign flows"],
        "bullish_conditions": "BI rate stable/falling, Rupiah strong, credit demand growing",
        "bearish_conditions": "BI rate rising, Rupiah weakening, bad loan concerns rising",
        "macro_tickers": {"Rupiah": "IDR=X", "EM flows": "EEM"},
    },
    "Coal": {
        "drivers": ["Newcastle coal price", "China demand", "export quotas", "energy transition policy"],
        "bullish_conditions": "Coal price above $120, China industrial demand strong, no export restrictions",
        "bearish_conditions": "Coal price falling, China slowdown, global energy transition accelerating",
        "macro_tickers": {"Coal": "MTF=F", "China": "FXI"},
    },
    "Mining": {
        "drivers": ["Commodity prices (nickel/copper/gold)", "China demand", "export ban policy", "EV demand"],
        "bullish_conditions": "Commodity prices rising, China buying, EV demand strong, no export bans",
        "bearish_conditions": "Commodity prices falling, export restrictions, global recession fears",
        "macro_tickers": {"Nickel": "JJN", "Copper": "CPER", "Gold": "GC=F", "China": "FXI"},
    },
    "Consumer": {
        "drivers": ["Indonesian inflation", "consumer confidence", "middle class growth", "Rupiah for imports"],
        "bullish_conditions": "Low inflation, rising wages, strong consumer sentiment",
        "bearish_conditions": "High inflation eating purchasing power, weak Rupiah raising import costs",
        "macro_tickers": {"Rupiah": "IDR=X"},
    },
    "Telco": {
        "drivers": ["subscriber growth", "data usage", "5G rollout", "competition"],
        "bullish_conditions": "Data demand growing, 5G expansion, consolidation reducing competition",
        "bearish_conditions": "Price wars, regulatory pressure, slowing subscriber growth",
        "macro_tickers": {},
    },
    "Energy": {
        "drivers": ["oil price", "government subsidy policy", "production volumes"],
        "bullish_conditions": "Oil above $80, stable subsidies, production growing",
        "bearish_conditions": "Oil below $60, subsidy cuts, production decline",
        "macro_tickers": {"Oil": "CL=F"},
    },
    "Property": {
        "drivers": ["mortgage rates", "government housing programs", "BI rate", "urban migration"],
        "bullish_conditions": "Low rates, government stimulus, urbanization driving demand",
        "bearish_conditions": "High rates, oversupply, weak consumer confidence",
        "macro_tickers": {"Rupiah": "IDR=X"},
    },
    "Healthcare": {
        "drivers": ["healthcare spending", "aging population", "BPJS coverage expansion"],
        "bullish_conditions": "Growing middle class, BPJS expansion, aging demographics",
        "bearish_conditions": "Generic competition, pricing pressure, weak Rupiah raising import costs",
        "macro_tickers": {},
    },
    "Technology": {
        "drivers": ["digital adoption", "GMV growth", "path to profitability", "fintech penetration"],
        "bullish_conditions": "User growth, improving unit economics, digital payment expansion",
        "bearish_conditions": "Cash burn, competition, regulatory pressure on gig economy",
        "macro_tickers": {},
    },
    "Conglomerate": {
        "drivers": ["auto sales", "commodity exposure", "financial services", "agribusiness"],
        "bullish_conditions": "Auto market strong, commodities up, consumer spending healthy",
        "bearish_conditions": "Auto sales slowing, commodity downturn, weak consumer sentiment",
        "macro_tickers": {"Rupiah": "IDR=X"},
    },
}

# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Fetch P/E, dividend yield, analyst data from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "pe_ratio":          info.get("trailingPE"),
            "forward_pe":        info.get("forwardPE"),
            "dividend_yield":    info.get("dividendYield"),
            "payout_ratio":      info.get("payoutRatio"),
            "revenue_growth":    info.get("revenueGrowth"),
            "earnings_growth":   info.get("earningsGrowth"),
            "profit_margin":     info.get("profitMargins"),
            "roe":               info.get("returnOnEquity"),
            "debt_to_equity":    info.get("debtToEquity"),
            "current_ratio":     info.get("currentRatio"),
            "analyst_target":    info.get("targetMeanPrice"),
            "analyst_rec":       info.get("recommendationKey"),
            "num_analysts":      info.get("numberOfAnalystOpinions"),
            "market_cap":        info.get("marketCap"),
            "52w_high":          info.get("fiftyTwoWeekHigh"),
            "52w_low":           info.get("fiftyTwoWeekLow"),
            "description":       info.get("longBusinessSummary", "")[:300],
            "employees":         info.get("fullTimeEmployees"),
            "sector":            info.get("sector"),
        }
    except:
        return {}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(search_query: str, max_items: int = 8) -> list:
    """Fetch recent news from Google News RSS."""
    try:
        query = search_query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-ID&gl=ID&ceid=ID:en"
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:max_items]:
            pub = entry.get("published", "")
            try:
                pub_dt = datetime(*entry.published_parsed[:6])
                days_ago = (datetime.now() - pub_dt).days
                age = f"{days_ago}d ago" if days_ago > 0 else "today"
            except:
                age = "recent"
            news.append({
                "title": entry.get("title", ""),
                "link":  entry.get("link", ""),
                "age":   age,
                "source": entry.get("source", {}).get("title", ""),
            })
        return news
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_series(ticker: str) -> pd.Series:
    try:
        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if df.empty: return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].iloc[:, 0].dropna()
        return df["Close"].dropna()
    except:
        return pd.Series(dtype=float)

# ── Helper functions ───────────────────────────────────────────────────────────

def get_close(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()

def get_volume(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Volume"].iloc[:, 0].dropna()
    return df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

def dma(s, w):
    if len(s) < w: return float("nan")
    return float(s.rolling(w).mean().iloc[-1])

def roc(s, d):
    if len(s) < d + 1: return float("nan")
    return float((s.iloc[-1] / s.iloc[-d-1] - 1) * 100)

def latest(s):
    return float(s.iloc[-1]) if not s.empty else float("nan")

def compute_rsi(s: pd.Series, period: int = 14) -> float:
    if len(s) < period + 1: return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def pct_from_high(s: pd.Series, period: int = 252) -> float:
    if len(s) < 5: return float("nan")
    high = s.tail(period).max()
    return float((s.iloc[-1] / high - 1) * 100)

def signal_color(score):
    if np.isnan(score): return "—", "—", "#6b7280"
    if score >= 65: return "🟢", "BUY", "#22c55e"
    if score >= 45: return "🟡", "HOLD", "#f59e0b"
    return "🔴", "SELL/WAIT", "#ef4444"

def confidence_bar(score, label="", color="#3b82f6"):
    if np.isnan(score): score = 0
    filled = int(score / 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    return f"`{bar}` **{score:.0f}%** {label}"

# ══════════════════════════════════════════════════════════════════════════════
# LAYER SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def score_layer1_macro(idr_data, idx_data, em_data) -> tuple:
    """Layer 1: Indonesia macro environment. Returns (score, details)."""
    score = 50.0
    details = []
    weights = []

    # IDX trend
    idx_close = get_close(idx_data)
    idx_1m = roc(idx_close, 21)
    idx_50 = dma(idx_close, 50)
    idx_now = latest(idx_close)

    if not np.isnan(idx_now) and not np.isnan(idx_50):
        if idx_now > idx_50:
            weights.append((75, 1.5, "IDX above 50DMA — market uptrend"))
        else:
            weights.append((25, 1.5, "IDX below 50DMA — market downtrend"))

    if not np.isnan(idx_1m):
        if idx_1m > 3:   weights.append((80, 1.0, f"IDX up {idx_1m:.1f}% this month — strong momentum"))
        elif idx_1m > 0: weights.append((60, 1.0, f"IDX up {idx_1m:.1f}% this month — mild positive"))
        elif idx_1m > -3:weights.append((40, 1.0, f"IDX down {idx_1m:.1f}% this month — mild negative"))
        else:            weights.append((20, 1.0, f"IDX down {idx_1m:.1f}% this month — weak market"))

    # Rupiah
    idr_close = get_close(idr_data)
    idr_1m = roc(idr_close, 21)
    idr_1w = roc(idr_close, 5)
    if not np.isnan(idr_1m):
        if idr_1m < -2:   weights.append((85, 1.5, f"Rupiah strengthening {abs(idr_1m):.1f}% — foreign money flowing in"))
        elif idr_1m < 0:  weights.append((65, 1.5, f"Rupiah slightly stronger {abs(idr_1m):.1f}% — stable"))
        elif idr_1m < 2:  weights.append((45, 1.5, f"Rupiah weakening {idr_1m:.1f}% — mild pressure"))
        else:             weights.append((20, 1.5, f"Rupiah weakening sharply {idr_1m:.1f}% — foreign selling risk"))

    # EM flows
    em_close = get_close(em_data)
    em_1m = roc(em_close, 21)
    if not np.isnan(em_1m):
        if em_1m > 2:    weights.append((75, 0.8, f"Emerging markets up {em_1m:.1f}% — global risk-on"))
        elif em_1m > 0:  weights.append((58, 0.8, f"EM markets slightly positive — neutral flows"))
        else:            weights.append((35, 0.8, f"EM markets down {em_1m:.1f}% — foreign outflows"))

    if weights:
        total_w = sum(w for _, w, _ in weights)
        score = sum(s * w for s, w, _ in weights) / total_w
        details = [d for _, _, d in weights]

    return max(0, min(100, score)), details


def score_layer2_sector(sector: str, sector_context: dict) -> tuple:
    """Layer 2: Sector-specific macro. Returns (score, details)."""
    score = 50.0
    details = []
    weights = []

    ctx = sector_context.get(sector, {})
    macro_tickers = ctx.get("macro_tickers", {})

    for name, ticker in macro_tickers.items():
        s = fetch_series(ticker)
        if s.empty: continue
        r1m = roc(s, 21)
        r1w = roc(s, 5)
        p = latest(s)

        if np.isnan(r1m): continue

        # Invert for Rupiah and US yields (rising = bad for Indo)
        if name in ["Rupiah"]:
            adj = -r1m  # falling USD/IDR = Rupiah strengthening = good
        else:
            adj = r1m

        if adj > 3:    weights.append((80, 1.0, f"{name}: +{r1m:.1f}% this month — strongly supportive"))
        elif adj > 0:  weights.append((62, 1.0, f"{name}: +{r1m:.1f}% this month — mildly supportive"))
        elif adj > -3: weights.append((40, 1.0, f"{name}: {r1m:.1f}% this month — mild headwind"))
        else:          weights.append((20, 1.0, f"{name}: {r1m:.1f}% this month — significant headwind"))

    if weights:
        total_w = sum(w for _, w, _ in weights)
        score = sum(s * w for s, w, _ in weights) / total_w
        details = [d for _, _, d in weights]
    else:
        details = [f"No real-time sector data available for {sector}"]

    # Add sector context
    details.append(f"Bullish if: {ctx.get('bullish_conditions', 'N/A')}")
    details.append(f"Bearish if: {ctx.get('bearish_conditions', 'N/A')}")

    return max(0, min(100, score)), details


def score_layer3_catalysts(news: list, sector: str) -> tuple:
    """Layer 3: News sentiment + catalyst scoring. Returns (score, details)."""
    if not news:
        return 50.0, ["No recent news found — neutral assumption"]

    score = 50.0
    details = []

    # Simple keyword scoring
    BULLISH_WORDS = [
        "profit", "growth", "record", "expansion", "investment", "dividend",
        "upgrade", "beat", "strong", "increase", "acquisition", "partnership",
        "contract", "approval", "launch", "positive", "recovery", "surge",
        "rises", "gains", "up", "buy", "outperform"
    ]
    BEARISH_WORDS = [
        "loss", "decline", "fall", "drop", "cut", "layoff", "fine", "ban",
        "risk", "concern", "downgrade", "miss", "weak", "debt", "default",
        "lawsuit", "investigation", "corruption", "sell", "underperform",
        "warning", "negative", "slump", "crash", "disappoints"
    ]

    bull_count = 0
    bear_count = 0
    scored_news = []

    for item in news:
        title_lower = item["title"].lower()
        b_hits = sum(1 for w in BULLISH_WORDS if w in title_lower)
        be_hits = sum(1 for w in BEARISH_WORDS if w in title_lower)

        if b_hits > be_hits:
            sentiment = "🟢"
            bull_count += 1
        elif be_hits > b_hits:
            sentiment = "🔴"
            bear_count += 1
        else:
            sentiment = "🟡"

        scored_news.append((sentiment, item["title"], item["age"], item.get("source", "")))

    total = bull_count + bear_count
    if total > 0:
        bull_ratio = bull_count / (len(news))
        bear_ratio = bear_count / (len(news))
        score = 50 + (bull_ratio - bear_ratio) * 40
    else:
        score = 50.0

    details = [f"Analysed {len(news)} recent news articles"]
    details.append(f"🟢 Positive: {bull_count} | 🔴 Negative: {bear_count} | 🟡 Neutral: {len(news)-bull_count-bear_count}")

    return max(20, min(85, score)), scored_news


def score_layer4_fundamentals(fund: dict, sector: str) -> tuple:
    """Layer 4: Valuation + fundamentals scoring. Returns (score, details)."""
    score = 50.0
    details = []
    weights = []

    # P/E ratio scoring (sector-adjusted)
    SECTOR_PE = {
        "Banking": 12, "Coal": 8, "Mining": 15, "Consumer": 25,
        "Telco": 18, "Energy": 12, "Property": 15, "Healthcare": 25,
        "Technology": 50, "Conglomerate": 15,
    }
    fair_pe = SECTOR_PE.get(sector, 15)
    pe = fund.get("pe_ratio")
    if pe and not np.isnan(pe) and pe > 0:
        if pe < fair_pe * 0.7:
            weights.append((85, 1.5, f"P/E {pe:.1f}x — cheap vs sector avg ({fair_pe}x) ✅"))
        elif pe < fair_pe:
            weights.append((68, 1.5, f"P/E {pe:.1f}x — fair value vs sector avg ({fair_pe}x) ✅"))
        elif pe < fair_pe * 1.3:
            weights.append((48, 1.5, f"P/E {pe:.1f}x — slightly expensive vs sector avg ({fair_pe}x)"))
        else:
            weights.append((25, 1.5, f"P/E {pe:.1f}x — expensive vs sector avg ({fair_pe}x) ⚠️"))

    # Dividend yield
    div = fund.get("dividend_yield")
    if div and not np.isnan(div):
        div_pct = div * 100
        if div_pct > 5:    weights.append((85, 1.0, f"Dividend yield {div_pct:.1f}% — excellent income ✅"))
        elif div_pct > 3:  weights.append((70, 1.0, f"Dividend yield {div_pct:.1f}% — good income ✅"))
        elif div_pct > 1:  weights.append((55, 1.0, f"Dividend yield {div_pct:.1f}% — modest income"))
        else:              weights.append((40, 1.0, f"Dividend yield {div_pct:.1f}% — low income"))

    # Revenue growth
    rev_g = fund.get("revenue_growth")
    if rev_g and not np.isnan(rev_g):
        rev_pct = rev_g * 100
        if rev_pct > 15:   weights.append((85, 1.2, f"Revenue growing {rev_pct:.1f}% YoY — strong ✅"))
        elif rev_pct > 5:  weights.append((68, 1.2, f"Revenue growing {rev_pct:.1f}% YoY — healthy"))
        elif rev_pct > 0:  weights.append((52, 1.2, f"Revenue growing {rev_pct:.1f}% YoY — slow"))
        else:              weights.append((25, 1.2, f"Revenue declining {rev_pct:.1f}% YoY ⚠️"))

    # Earnings growth
    earn_g = fund.get("earnings_growth")
    if earn_g and not np.isnan(earn_g):
        earn_pct = earn_g * 100
        if earn_pct > 20:  weights.append((88, 1.2, f"Earnings growing {earn_pct:.1f}% YoY — excellent ✅"))
        elif earn_pct > 5: weights.append((68, 1.2, f"Earnings growing {earn_pct:.1f}% YoY — good"))
        elif earn_pct > 0: weights.append((52, 1.2, f"Earnings growing {earn_pct:.1f}% YoY — slow"))
        else:              weights.append((22, 1.2, f"Earnings declining {earn_pct:.1f}% YoY ⚠️"))

    # ROE
    roe = fund.get("roe")
    if roe and not np.isnan(roe):
        roe_pct = roe * 100
        if roe_pct > 20:   weights.append((85, 0.8, f"ROE {roe_pct:.1f}% — excellent returns ✅"))
        elif roe_pct > 12: weights.append((65, 0.8, f"ROE {roe_pct:.1f}% — good returns"))
        elif roe_pct > 5:  weights.append((48, 0.8, f"ROE {roe_pct:.1f}% — modest returns"))
        else:              weights.append((25, 0.8, f"ROE {roe_pct:.1f}% — poor returns ⚠️"))

    # Analyst recommendation
    rec = fund.get("analyst_rec", "")
    target = fund.get("analyst_target")
    n_analysts = fund.get("num_analysts", 0)
    if rec:
        rec_scores = {"strong_buy": 90, "buy": 75, "hold": 50, "underperform": 30, "sell": 15}
        rec_score = rec_scores.get(rec.lower().replace(" ", "_"), 50)
        n_str = f"({n_analysts} analysts)" if n_analysts else ""
        weights.append((rec_score, 1.0, f"Analyst consensus: {rec.upper()} {n_str}"))
        if target:
            details.append(f"Analyst price target: Rp {target:,.0f}")

    if weights:
        total_w = sum(w for _, w, _ in weights)
        score = sum(s * w for s, w, _ in weights) / total_w
        details = [d for _, _, d in weights] + details
    else:
        score = 50.0
        details = ["Fundamental data not available from Yahoo Finance for this stock"]

    return max(0, min(100, score)), details


def score_layer5_technical(close: pd.Series, volume: pd.Series) -> tuple:
    """Layer 5: Technical analysis. Returns (score, details)."""
    if close.empty or len(close) < 20:
        return 50.0, ["Insufficient price data for technical analysis"]

    score = 50.0
    details = []
    weights = []

    p = latest(close)

    # Moving averages
    for w, label, weight in [(20, "20DMA", 1.0), (50, "50DMA", 1.5), (200, "200DMA", 2.0)]:
        ma = dma(close, w)
        if not np.isnan(ma):
            pct_diff = (p / ma - 1) * 100
            if p > ma:
                s = min(85, 60 + pct_diff * 2)
                weights.append((s, weight, f"Above {label} (+{pct_diff:.1f}%) ✅"))
            else:
                s = max(15, 45 + pct_diff * 2)
                weights.append((s, weight, f"Below {label} ({pct_diff:.1f}%) ⚠️"))

    # RSI
    rsi = compute_rsi(close)
    if not np.isnan(rsi):
        if rsi < 30:       weights.append((80, 1.0, f"RSI {rsi:.0f} — oversold, potential bounce ✅"))
        elif rsi < 45:     weights.append((65, 1.0, f"RSI {rsi:.0f} — approaching oversold, mild bullish"))
        elif rsi < 60:     weights.append((55, 1.0, f"RSI {rsi:.0f} — neutral"))
        elif rsi < 75:     weights.append((45, 1.0, f"RSI {rsi:.0f} — approaching overbought"))
        else:              weights.append((25, 1.0, f"RSI {rsi:.0f} — overbought, caution ⚠️"))

    # Momentum
    r1w  = roc(close, 5)
    r1m  = roc(close, 21)
    r3m  = roc(close, 63)
    if not np.isnan(r1m):
        if r1m > 5:    weights.append((78, 0.8, f"1-month momentum: +{r1m:.1f}% — strong"))
        elif r1m > 0:  weights.append((60, 0.8, f"1-month momentum: +{r1m:.1f}% — positive"))
        elif r1m > -5: weights.append((42, 0.8, f"1-month momentum: {r1m:.1f}% — weak"))
        else:          weights.append((22, 0.8, f"1-month momentum: {r1m:.1f}% — very weak ⚠️"))

    # Distance from 52-week high
    dist_high = pct_from_high(close, 252)
    if not np.isnan(dist_high):
        if dist_high > -5:    weights.append((80, 0.6, f"Near 52-week high ({dist_high:.1f}%) — strong trend"))
        elif dist_high > -15: weights.append((60, 0.6, f"{dist_high:.1f}% from 52-week high — moderate"))
        elif dist_high > -30: weights.append((40, 0.6, f"{dist_high:.1f}% from 52-week high — weak"))
        else:                 weights.append((20, 0.6, f"{dist_high:.1f}% from 52-week high — very weak ⚠️"))

    # Volume trend
    if not volume.empty and len(volume) >= 20:
        vol_recent = volume.tail(5).mean()
        vol_avg    = volume.tail(20).mean()
        if not np.isnan(vol_recent) and not np.isnan(vol_avg) and vol_avg > 0:
            vol_ratio = vol_recent / vol_avg
            if vol_ratio > 1.5 and r1m and r1m > 0:
                weights.append((75, 0.5, f"Volume {vol_ratio:.1f}x average on up move — institutional buying"))
            elif vol_ratio > 1.5 and r1m and r1m < 0:
                weights.append((25, 0.5, f"Volume {vol_ratio:.1f}x average on down move — distribution ⚠️"))
            else:
                weights.append((50, 0.5, f"Volume {vol_ratio:.1f}x average — normal"))

    if weights:
        total_w = sum(w for _, w, _ in weights)
        score = sum(s * w for s, w, _ in weights) / total_w
        details = [d for _, _, d in weights]

    # Add key stats
    if not np.isnan(r1w):  details.append(f"1 week: {r1w:+.1f}%")
    if not np.isnan(r1m):  details.append(f"1 month: {r1m:+.1f}%")
    if not np.isnan(r3m):  details.append(f"3 months: {r3m:+.1f}%")
    if not np.isnan(rsi):  details.append(f"RSI: {rsi:.0f}")

    return max(0, min(100, score)), details


def compute_final_score(l1, l2, l3, l4, l5) -> float:
    """Weighted average of all 5 layers."""
    weights = [0.20, 0.20, 0.15, 0.25, 0.20]
    scores  = [l1, l2, l3, l4, l5]
    valid = [(s, w) for s, w in zip(scores, weights) if not np.isnan(s)]
    if not valid: return 50.0
    total_w = sum(w for _, w in valid)
    return sum(s * w for s, w in valid) / total_w


def make_chart(close: pd.Series, name: str, score: float, height: int = 320) -> go.Figure:
    color = "#22c55e" if score >= 65 else "#f59e0b" if score >= 45 else "#ef4444"
    fig = go.Figure()
    sp = close.tail(252)
    fig.add_trace(go.Scatter(x=sp.index, y=sp.values, name=name,
                              line=dict(color=color, width=2)))
    for w, c, label in [(20, "#94a3b8", "20DMA"), (50, "#f59e0b", "50DMA"), (200, "#ef4444", "200DMA")]:
        if len(close) >= w:
            ma = close.rolling(w).mean().tail(252)
            fig.add_trace(go.Scatter(x=ma.index, y=ma.values, name=label,
                                      line=dict(color=c, width=1, dash="dot")))
    fig.update_layout(height=height, margin=dict(l=10,r=10,t=10,b=10),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), showlegend=True,
                      legend=dict(orientation="h", y=1.08),
                      xaxis=dict(gridcolor="#1f2937"),
                      yaxis=dict(gridcolor="#1f2937"))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🇮🇩 Indo Deep Analysis")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    view = st.radio("View", ["🔍 Deep Stock Analysis", "📊 Market Overview", "🏦 All Stocks Screener"])
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M')}")
    st.caption("⚠️ Not financial advice.")
    with st.expander("📖 How the 5 layers work"):
        st.markdown("""
**Layer 1 — Indo Macro (20%)**
IDX trend, Rupiah, EM flows

**Layer 2 — Sector (20%)**
Commodity prices, sector-specific drivers

**Layer 3 — Catalysts/News (15%)**
Recent news sentiment analysis

**Layer 4 — Fundamentals (25%)**
P/E, dividends, earnings growth, ROE, analyst targets

**Layer 5 — Technical (20%)**
Moving averages, RSI, momentum, volume

**Final signal:**
- 🟢 65%+ = BUY
- 🟡 45-64% = HOLD  
- 🔴 <45% = SELL/WAIT
        """)

# ── Fetch shared macro data ───────────────────────────────────────────────────
with st.spinner("Loading macro data..."):
    idx_data = fetch_price("^JKSE", "6mo")
    idr_data = fetch_price("IDR=X", "6mo")
    em_data  = fetch_price("EEM",   "6mo")

# ══════════════════════════════════════════════════════════════════════════════
# DEEP STOCK ANALYSIS VIEW
# ══════════════════════════════════════════════════════════════════════════════
if view == "🔍 Deep Stock Analysis":
    st.title("🔍 Deep Stock Analysis")
    st.caption("5-layer confidence scoring system")

    selected = st.selectbox("Select a stock to analyse", list(STOCKS.keys()))
    stock_info = STOCKS[selected]
    ticker = stock_info["ticker"]
    sector = stock_info["sector"]
    search = stock_info["search"]

    st.markdown("---")

    with st.spinner(f"Running full analysis on {selected}..."):
        # Fetch all data
        price_df   = fetch_price(ticker, "1y")
        close      = get_close(price_df)
        volume     = get_volume(price_df)
        fund       = fetch_fundamentals(ticker)
        news       = fetch_news(search)
        macro_news = fetch_news("Indonesia economy Bank Indonesia")

        # Score all 5 layers
        l1_score, l1_details = score_layer1_macro(idr_data, idx_data, em_data)
        l2_score, l2_details = score_layer2_sector(sector, SECTOR_CONTEXT)
        l3_score, l3_news    = score_layer3_catalysts(news, sector)
        l4_score, l4_details = score_layer4_fundamentals(fund, sector)
        l5_score, l5_details = score_layer5_technical(close, volume)
        final_score          = compute_final_score(l1_score, l2_score, l3_score, l4_score, l5_score)

    em, sl, color = signal_color(final_score)
    p = latest(close)
    r1m = roc(close, 21)

    # ── TOP SUMMARY ────────────────────────────────────────────────────────────
    st.markdown(f"## {em} {selected}")
    if not np.isnan(p):
        st.metric("Current Price", f"Rp {p:,.0f}",
                  delta=f"{r1m:+.1f}% (1 month)" if not np.isnan(r1m) else None)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### Overall Signal: **{sl}** — {final_score:.0f}% confidence")
        st.progress(int(final_score) / 100)
        st.markdown("**5-Layer Breakdown:**")
        st.markdown(confidence_bar(l1_score, "Macro Environment"))
        st.markdown(confidence_bar(l2_score, f"{sector} Sector"))
        st.markdown(confidence_bar(l3_score, "News & Catalysts"))
        st.markdown(confidence_bar(l4_score, "Fundamentals"))
        st.markdown(confidence_bar(l5_score, "Technical"))

    with col2:
        # Key fundamentals
        st.markdown("**Key Numbers:**")
        pe = fund.get("pe_ratio")
        div = fund.get("dividend_yield")
        rev_g = fund.get("revenue_growth")
        earn_g = fund.get("earnings_growth")
        roe = fund.get("roe")
        rec = fund.get("analyst_rec")
        target = fund.get("analyst_target")

        if pe:    st.metric("P/E Ratio", f"{pe:.1f}x")
        if div:   st.metric("Dividend Yield", f"{div*100:.1f}%")
        if rev_g: st.metric("Revenue Growth", f"{rev_g*100:+.1f}%")
        if earn_g:st.metric("Earnings Growth", f"{earn_g*100:+.1f}%")
        if roe:   st.metric("ROE", f"{roe*100:.1f}%")
        if rec:   st.metric("Analyst View", rec.upper())
        if target:st.metric("Price Target", f"Rp {target:,.0f}")

    # ── CHART ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    if not close.empty:
        st.plotly_chart(make_chart(close, selected, final_score), use_container_width=True)

    # ── LAYER DETAILS ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Full Layer Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"🌏 Layer 1: Macro ({l1_score:.0f}%)",
        f"🏭 Layer 2: Sector ({l2_score:.0f}%)",
        f"📰 Layer 3: News ({l3_score:.0f}%)",
        f"💰 Layer 4: Fundamentals ({l4_score:.0f}%)",
        f"📈 Layer 5: Technical ({l5_score:.0f}%)",
    ])

    with tab1:
        st.markdown(f"**Indonesia Macro Score: {l1_score:.0f}%**")
        st.progress(int(l1_score) / 100)
        st.markdown("**What this measures:** IDX trend, Rupiah strength, emerging market flows")
        for d in l1_details:
            st.markdown(f"· {d}")
        # Macro news
        if macro_news:
            st.markdown("**Recent Indonesia Macro News:**")
            for item in macro_news[:4]:
                st.markdown(f"· [{item['title']}]({item['link']}) — *{item['age']}*")

    with tab2:
        st.markdown(f"**{sector} Sector Score: {l2_score:.0f}%**")
        st.progress(int(l2_score) / 100)
        ctx = SECTOR_CONTEXT.get(sector, {})
        st.markdown(f"**Key drivers:** {', '.join(ctx.get('drivers', []))}")
        for d in l2_details:
            st.markdown(f"· {d}")
        # Sector news
        sector_news = fetch_news(f"{sector} Indonesia stocks 2025")
        if sector_news:
            st.markdown(f"**Recent {sector} News:**")
            for item in sector_news[:4]:
                st.markdown(f"· [{item['title']}]({item['link']}) — *{item['age']}*")

    with tab3:
        st.markdown(f"**News & Catalyst Score: {l3_score:.0f}%**")
        st.progress(int(l3_score) / 100)
        st.markdown(f"**Searched for:** *{search}*")
        if l3_news and isinstance(l3_news[0], tuple):
            st.markdown("**Recent News Headlines:**")
            for sentiment, title, age, source in l3_news:
                src_str = f" — {source}" if source else ""
                st.markdown(f"{sentiment} {title} *(({age}){src_str})*")
        else:
            for d in l3_news:
                st.markdown(f"· {d}")

    with tab4:
        st.markdown(f"**Fundamentals Score: {l4_score:.0f}%**")
        st.progress(int(l4_score) / 100)
        for d in l4_details:
            st.markdown(f"· {d}")

        # Detailed fundamentals table
        desc = fund.get("description", "")
        if desc:
            st.markdown("**About the company:**")
            st.caption(desc + "...")

        # 52W range
        high52 = fund.get("52w_high")
        low52  = fund.get("52w_low")
        if high52 and low52:
            st.markdown(f"**52-week range:** Rp {low52:,.0f} — Rp {high52:,.0f}")
            if not np.isnan(p):
                pct_in_range = (p - low52) / (high52 - low52) * 100
                st.caption(f"Currently at {pct_in_range:.0f}% of 52-week range")

    with tab5:
        st.markdown(f"**Technical Score: {l5_score:.0f}%**")
        st.progress(int(l5_score) / 100)
        for d in l5_details:
            st.markdown(f"· {d}")

    st.markdown("---")

    # ── VERDICT ────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Investment Verdict")

    # Generate reasons
    all_scores = [
        ("Macro environment", l1_score),
        (f"{sector} sector", l2_score),
        ("Recent news", l3_score),
        ("Fundamentals", l4_score),
        ("Technical trend", l5_score),
    ]
    strong_bull = [n for n, s in all_scores if s >= 65]
    strong_bear = [n for n, s in all_scores if s < 40]

    if final_score >= 65:
        st.success(f"🟢 **BUY** — {final_score:.0f}% confidence")
        if strong_bull:
            st.markdown(f"**Reasons to buy:** {', '.join(strong_bull)} are all supportive")
        if strong_bear:
            st.markdown(f"**Watch out for:** {', '.join(strong_bear)} — these are risks")
    elif final_score >= 45:
        st.warning(f"🟡 **HOLD** — {final_score:.0f}% confidence")
        st.markdown("Mixed signals. Don't add new money, but no need to sell existing position.")
        if strong_bull:
            st.markdown(f"**Positives:** {', '.join(strong_bull)}")
        if strong_bear:
            st.markdown(f"**Concerns:** {', '.join(strong_bear)}")
    else:
        st.error(f"🔴 **SELL/WAIT** — {final_score:.0f}% confidence")
        if strong_bear:
            st.markdown(f"**Main concerns:** {', '.join(strong_bear)}")
        st.markdown("Consider waiting for better entry conditions before adding.")

    st.caption("⚠️ This analysis is algorithmic and informational only. Not financial advice. Always do your own research.")

# ══════════════════════════════════════════════════════════════════════════════
# MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif view == "📊 Market Overview":
    st.title("📊 Indonesia Market Overview")

    idx_close = get_close(idx_data)
    idr_close = get_close(idr_data)

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_now = latest(idx_close)
        idx_1m  = roc(idx_close, 21)
        st.metric("IDX Composite", f"{idx_now:,.0f}" if not np.isnan(idx_now) else "—",
                  delta=f"{idx_1m:+.1f}% (1mo)" if not np.isnan(idx_1m) else None)
    with col2:
        idr_now = latest(idr_close)
        idr_1m  = roc(idr_close, 21)
        st.metric("USD/IDR", f"{idr_now:,.0f}" if not np.isnan(idr_now) else "—",
                  delta=f"{idr_1m:+.1f}% (1mo)" if not np.isnan(idr_1m) else None,
                  delta_color="inverse")
    with col3:
        l1_score, _ = score_layer1_macro(idr_data, idx_data, em_data)
        e, sl, _ = signal_color(l1_score)
        st.metric("Macro Health", f"{l1_score:.0f}%", delta=f"{e} {sl}")

    col1, col2 = st.columns(2)
    with col1:
        if not idx_close.empty:
            st.plotly_chart(make_chart(idx_close, "IDX", l1_score, 250), use_container_width=True)
    with col2:
        if not idr_close.empty:
            idr_score = 100 - roc(idr_close, 21) * 5 if not np.isnan(roc(idr_close, 21)) else 50
            st.plotly_chart(make_chart(idr_close, "USD/IDR", 100 - idr_score, 250), use_container_width=True)

    # Macro news
    st.markdown("### 📰 Latest Indonesia Economic News")
    macro_news = fetch_news("Indonesia economy investment stocks 2025", 10)
    if macro_news:
        for item in macro_news:
            st.markdown(f"· [{item['title']}]({item['link']}) — *{item['age']}*")
    else:
        st.info("News feed not available in this environment. Works on your local machine.")

# ══════════════════════════════════════════════════════════════════════════════
# ALL STOCKS SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif view == "🏦 All Stocks Screener":
    st.title("🏦 Stock Screener")
    st.caption("Quick scores for all stocks. Click Deep Analysis for full breakdown.")

    sector_filter = st.selectbox("Filter by sector",
        ["All"] + sorted(list(set(v["sector"] for v in STOCKS.values()))))

    rows = []
    progress = st.progress(0)
    stocks_list = list(STOCKS.items())

    for i, (name, info) in enumerate(stocks_list):
        progress.progress((i + 1) / len(stocks_list))
        if sector_filter != "All" and info["sector"] != sector_filter:
            continue

        ticker = info["ticker"]
        sector = info["sector"]
        close = fetch_series(ticker)
        p     = latest(close)
        r1m   = roc(close, 21)
        r1w   = roc(close, 5)
        sc5   = 50.0  # technical only for screener speed

        if not close.empty and len(close) >= 10:
            sc5_val, _ = score_layer5_technical(close, pd.Series(dtype=float))
            sc5 = sc5_val

        l1, _ = score_layer1_macro(idr_data, idx_data, em_data)
        quick_score = (l1 * 0.25 + sc5 * 0.75)  # fast approximation
        e2, sl2, _ = signal_color(quick_score)

        rows.append({
            "Stock":    name,
            "Sector":   sector,
            "Price":    f"Rp {p:,.0f}" if not np.isnan(p) else "—",
            "1 Week":   f"{r1w:+.1f}%" if not np.isnan(r1w) else "—",
            "1 Month":  f"{r1m:+.1f}%" if not np.isnan(r1m) else "—",
            "Tech Score":f"{sc5:.0f}%",
            "Signal":   f"{e2} {sl2}",
            "_score":   quick_score,
        })

    progress.empty()

    df = pd.DataFrame(rows).sort_values("_score", ascending=False).drop(columns=["_score"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("💡 For full 5-layer analysis including fundamentals and news, use 'Deep Stock Analysis'")
