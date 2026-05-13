"""
Indonesian Market Dashboard — AI Analysis Edition
Powered by Google Gemini (free tier)
Run with: python -m streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import feedparser
from datetime import datetime


st.set_page_config(page_title="Indo AI Dashboard", page_icon="🇮🇩", layout="wide")

# ── Gemini API Key ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

# ── Stock Universe ─────────────────────────────────────────────────────────────
STOCKS = {
    "BCA (BBCA)":          {"ticker": "BBCA.JK", "sector": "Banking",     "search": "BCA Bank Central Asia Indonesia"},
    "BRI (BBRI)":          {"ticker": "BBRI.JK", "sector": "Banking",     "search": "BRI Bank Rakyat Indonesia"},
    "Mandiri (BMRI)":      {"ticker": "BMRI.JK", "sector": "Banking",     "search": "Bank Mandiri Indonesia"},
    "BNI (BBNI)":          {"ticker": "BBNI.JK", "sector": "Banking",     "search": "BNI Bank Negara Indonesia"},
    "BTN (BBTN)":          {"ticker": "BBTN.JK", "sector": "Banking",     "search": "BTN Bank Tabungan Negara"},
    "Bayan Coal (BYAN)":   {"ticker": "BYAN.JK", "sector": "Coal",        "search": "Bayan Resources coal Indonesia"},
    "Adaro (ADRO)":        {"ticker": "ADRO.JK", "sector": "Coal",        "search": "Adaro Energy coal Indonesia"},
    "Bukit Asam (PTBA)":   {"ticker": "PTBA.JK", "sector": "Coal",        "search": "Bukit Asam coal Indonesia"},
    "Harum Energy (HRUM)": {"ticker": "HRUM.JK", "sector": "Coal",        "search": "Harum Energy coal Indonesia"},
    "Vale Indonesia (INCO)":{"ticker":"INCO.JK",  "sector": "Mining",     "search": "Vale Indonesia nickel mining"},
    "Aneka Tambang (ANTM)":{"ticker": "ANTM.JK", "sector": "Mining",      "search": "Antam gold nickel Indonesia"},
    "Merdeka Copper (MDKA)":{"ticker":"MDKA.JK",  "sector": "Mining",     "search": "Merdeka Copper Gold Indonesia"},
    "Indofood CB (ICBP)":  {"ticker": "ICBP.JK", "sector": "Consumer",    "search": "Indofood CBP consumer Indonesia"},
    "Indofood (INDF)":     {"ticker": "INDF.JK", "sector": "Consumer",    "search": "Indofood consumer Indonesia"},
    "Unilever (UNVR)":     {"ticker": "UNVR.JK", "sector": "Consumer",    "search": "Unilever Indonesia"},
    "Kalbe Farma (KLBF)":  {"ticker": "KLBF.JK", "sector": "Healthcare",  "search": "Kalbe Farma pharmaceutical Indonesia"},
    "Telkom (TLKM)":       {"ticker": "TLKM.JK", "sector": "Telco",       "search": "Telkom Indonesia telecommunications"},
    "Indosat (ISAT)":      {"ticker": "ISAT.JK", "sector": "Telco",       "search": "Indosat Ooredoo Indonesia"},
    "XL Axiata (EXCL)":    {"ticker": "EXCL.JK", "sector": "Telco",       "search": "XL Axiata Indonesia"},
    "Medco Energy (MEDC)": {"ticker": "MEDC.JK", "sector": "Energy",      "search": "Medco Energy oil gas Indonesia"},
    "Ciputra (CTRA)":      {"ticker": "CTRA.JK", "sector": "Property",    "search": "Ciputra property Indonesia"},
    "Bumi Serpong (BSDE)": {"ticker": "BSDE.JK", "sector": "Property",    "search": "Bumi Serpong Damai property Indonesia"},
    "Astra (ASII)":        {"ticker": "ASII.JK", "sector": "Conglomerate","search": "Astra International Indonesia"},
    "GOTO":                {"ticker": "GOTO.JK", "sector": "Technology",  "search": "GOTO Gojek Tokopedia Indonesia"},
}

SECTOR_PE = {
    "Banking": 12, "Coal": 8, "Mining": 15, "Consumer": 25,
    "Telco": 18, "Energy": 12, "Property": 15, "Healthcare": 25,
    "Technology": 50, "Conglomerate": 15,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_series(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty: return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].iloc[:, 0].dropna()
        return df["Close"].dropna()
    except:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "pe_ratio":       info.get("trailingPE"),
            "forward_pe":     info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth":info.get("earningsGrowth"),
            "profit_margin":  info.get("profitMargins"),
            "roe":            info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_rec":    info.get("recommendationKey"),
            "num_analysts":   info.get("numberOfAnalystOpinions"),
            "market_cap":     info.get("marketCap"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "description":    info.get("longBusinessSummary", "")[:500],
        }
    except:
        return {}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(query, max_items=8):
    try:
        q = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-ID&gl=ID&ceid=ID:en"
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:max_items]:
            try:
                pub_dt = datetime(*entry.published_parsed[:6])
                days_ago = (datetime.now() - pub_dt).days
                age = f"{days_ago}d ago" if days_ago > 0 else "today"
            except:
                age = "recent"
            news.append({
                "title":  entry.get("title", ""),
                "link":   entry.get("link", ""),
                "age":    age,
                "source": entry.get("source", {}).get("title", ""),
            })
        return news
    except:
        return []


def get_currency(ticker):
    """Return currency symbol based on ticker suffix."""
    t = ticker.upper()
    if t.endswith(".JK"): return "Rp", "IDR"
    if t.endswith(".T"): return "¥", "JPY"
    if t.endswith(".HK"): return "HK$", "HKD"
    if t.endswith(".KS") or t.endswith(".KQ"): return "₩", "KRW"
    if t.endswith(".SS") or t.endswith(".SZ"): return "¥", "CNY"
    if t.endswith(".AX"): return "A$", "AUD"
    if t.endswith(".L"): return "£", "GBP"
    if "BTC" in t or "ETH" in t or "XRP" in t or "SOL" in t or "-USD" in t: return "$", "USD"
    return "$", "USD"  # default US stocks

def get_close(df):
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()

def get_volume(df):
    if df.empty: return pd.Series(dtype=float)
    try:
        if isinstance(df.columns, pd.MultiIndex):
            return df["Volume"].iloc[:, 0].dropna()
        return df["Volume"].dropna()
    except:
        return pd.Series(dtype=float)

def dma(s, w):
    if len(s) < w: return float("nan")
    return float(s.rolling(w).mean().iloc[-1])

def roc(s, d):
    if len(s) < d + 1: return float("nan")
    return float((s.iloc[-1] / s.iloc[-d-1] - 1) * 100)

def latest(s):
    return float(s.iloc[-1]) if not s.empty else float("nan")

def compute_rsi(s, period=14):
    if len(s) < period + 1: return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return float((100 - (100 / (1 + rs))).iloc[-1])

def signal_color(score):
    if np.isnan(score): return "—", "—", "#6b7280"
    if score >= 65: return "🟢", "BUY", "#22c55e"
    if score >= 45: return "🟡", "HOLD", "#f59e0b"
    return "🔴", "SELL/WAIT", "#ef4444"

def score_technical(close, volume):
    if close.empty or len(close) < 20:
        return 50.0, []
    score = 50.0
    details = []
    weights = []
    p = latest(close)
    for w, label, wt in [(20,"20DMA",1.0),(50,"50DMA",1.5),(200,"200DMA",2.0)]:
        ma = dma(close, w)
        if not np.isnan(ma):
            pct = (p/ma-1)*100
            s = min(85, 60+pct*2) if p > ma else max(15, 45+pct*2)
            weights.append((s, wt, f"{'✅' if p>ma else '⚠️'} {'Above' if p>ma else 'Below'} {label} ({pct:+.1f}%)"))
    rsi = compute_rsi(close)
    if not np.isnan(rsi):
        if rsi < 30:    weights.append((80,1.0,f"✅ RSI {rsi:.0f} — oversold"))
        elif rsi < 45:  weights.append((65,1.0,f"RSI {rsi:.0f} — mild bullish"))
        elif rsi < 60:  weights.append((55,1.0,f"RSI {rsi:.0f} — neutral"))
        elif rsi < 75:  weights.append((45,1.0,f"RSI {rsi:.0f} — approaching overbought"))
        else:           weights.append((25,1.0,f"⚠️ RSI {rsi:.0f} — overbought"))
    r1m = roc(close,21)
    if not np.isnan(r1m):
        if r1m > 5:    weights.append((78,0.8,f"✅ 1M momentum: +{r1m:.1f}%"))
        elif r1m > 0:  weights.append((60,0.8,f"1M momentum: +{r1m:.1f}%"))
        elif r1m > -5: weights.append((42,0.8,f"1M momentum: {r1m:.1f}%"))
        else:          weights.append((22,0.8,f"⚠️ 1M momentum: {r1m:.1f}%"))
    if weights:
        tw = sum(w for _,w,_ in weights)
        score = sum(s*w for s,w,_ in weights)/tw
        details = [d for _,_,d in weights]
    return max(0,min(100,score)), details

def score_fundamentals(fund, sector):
    score = 50.0
    details = []
    weights = []
    fair_pe = SECTOR_PE.get(sector, 15)
    pe = fund.get("pe_ratio")
    if pe and not np.isnan(pe) and pe > 0:
        if pe < fair_pe*0.7:   weights.append((85,1.5,f"✅ P/E {pe:.1f}x — cheap (sector avg: {fair_pe}x)"))
        elif pe < fair_pe:     weights.append((68,1.5,f"✅ P/E {pe:.1f}x — fair value"))
        elif pe < fair_pe*1.3: weights.append((48,1.5,f"P/E {pe:.1f}x — slightly expensive"))
        else:                  weights.append((25,1.5,f"⚠️ P/E {pe:.1f}x — expensive"))
    div = fund.get("dividend_yield")
    if div and not np.isnan(div):
        d = div*100
        if d > 5:    weights.append((85,1.0,f"✅ Dividend {d:.1f}% — excellent"))
        elif d > 3:  weights.append((70,1.0,f"✅ Dividend {d:.1f}% — good"))
        elif d > 1:  weights.append((55,1.0,f"Dividend {d:.1f}% — modest"))
        else:        weights.append((40,1.0,f"Dividend {d:.1f}% — low"))
    eg = fund.get("earnings_growth")
    if eg and not np.isnan(eg):
        e = eg*100
        if e > 20:   weights.append((88,1.2,f"✅ Earnings growth {e:.1f}%"))
        elif e > 5:  weights.append((68,1.2,f"Earnings growth {e:.1f}%"))
        elif e > 0:  weights.append((52,1.2,f"Earnings growth {e:.1f}% — slow"))
        else:        weights.append((22,1.2,f"⚠️ Earnings declining {e:.1f}%"))
    roe = fund.get("roe")
    if roe and not np.isnan(roe):
        r = roe*100
        if r > 20:   weights.append((85,0.8,f"✅ ROE {r:.1f}% — excellent"))
        elif r > 12: weights.append((65,0.8,f"ROE {r:.1f}% — good"))
        elif r > 5:  weights.append((48,0.8,f"ROE {r:.1f}% — modest"))
        else:        weights.append((25,0.8,f"⚠️ ROE {r:.1f}% — poor"))
    rec = fund.get("analyst_rec","")
    if rec:
        rs = {"strong_buy":90,"buy":75,"hold":50,"underperform":30,"sell":15}
        weights.append((rs.get(rec.lower().replace(" ","_"),50),1.0,f"Analyst: {rec.upper()}"))
    if weights:
        tw = sum(w for _,w,_ in weights)
        score = sum(s*w for s,w,_ in weights)/tw
        details = [d for _,_,d in weights]
    else:
        details = ["Fundamental data not available"]
    return max(0,min(100,score)), details

# ── Gemini AI Analysis ────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def run_gemini_analysis(stock_name, ticker, sector, price, roc_1m, roc_3m,
                         rsi, above_200dma, pe, div_yield, earnings_growth,
                         analyst_rec, news_headlines, macro_context,
                         tech_score, fund_score):
    """Call Gemini API with the full analyst prompt framework."""
    try:
        # Build context summary
        curr_sym_ai, _ = get_currency(ticker)
        tech_summary = f"Price {curr_sym_ai} {price:,.2f}, 1M return {roc_1m:+.1f}%, 3M return {roc_3m:+.1f}%, RSI {rsi:.0f}, {'above' if above_200dma else 'below'} 200DMA"
        fund_summary = f"P/E {pe:.1f}x, Dividend yield {div_yield:.1f}%, Earnings growth {earnings_growth:+.1f}%, Analyst: {analyst_rec}"
        news_str = "\n".join([f"- {n['title']} ({n['age']})" for n in news_headlines[:6]]) if news_headlines else "No recent news available"

        prompt = f"""You are a neutral, institutional-grade market analyst writing for a cautious retail investor based in Indonesia.
Your job is to give a balanced, practical analysis. Be objective. No hype. No price targets. No guarantees.
Write in simple, clear English with headings and bullet points.

Target stock: {stock_name} ({ticker}) — {sector} sector

LIVE MARKET DATA:
- Technical: {tech_summary}
- Technical score: {tech_score:.0f}/100
- Fundamentals: {fund_summary}
- Fundamental score: {fund_score:.0f}/100
- Macro context: {macro_context}

IMPORTANT INSTRUCTIONS FOR MACRO ANALYSIS:
- If this is a US stock (no suffix or .US): focus on US Fed policy, US earnings cycle, US sector trends, S&P500/Nasdaq momentum, USD strength. Only mention Indonesia in the context of currency risk for the investor.
- If this is an Indonesian stock (.JK): focus on Bank Indonesia rate, Rupiah, IDX trend, commodity prices, domestic demand.
- If this is a Japanese stock (.T): focus on BOJ policy, Yen direction, Japan macro, domestic sector trends.
- If this is a Korean stock (.KS): focus on BOK policy, KRW, Samsung/chip cycle, export demand from China.
- If this is a Chinese/HK stock (.HK, .SS): focus on PBOC policy, China growth, property sector, regulatory environment.
- If this is crypto (-USD): focus on BTC dominance, macro risk appetite, on-chain signals, regulatory news.
- Always end Section 1 with a note on what this means specifically for an Indonesian retail investor holding this asset (currency risk, accessibility, liquidity).

RECENT NEWS:
{news_str}

IMPORTANT: Start your response with exactly this format (fill in the numbers):

CONFIDENCE_SCORES:
BUY: [0-100]
HOLD: [0-100]  
SELL: [0-100]
OVERALL: [BUY/HOLD/SELL]
ONE_LINE: [one sentence summary]
END_SCORES

Then write the full analysis below.

---

Please write a complete investment analysis with these 7 sections:

---
## 1. Market Mindset Check (Macro Backdrop)
Explain the current Indonesia market environment for a cautious retail trader.
Focus on: sentiment, Rupiah impact, BI rate direction, foreign flows, commodity prices if relevant to this sector.
**What to watch next:** (3-4 bullets)

---
## 2. {stock_name} in One Page
Explain this company as if I've never heard of it:
- What it does (plain language)
- How it makes money
- Top 5 drivers of the stock price
- Main competitors
- The ONE thing that matters most right now
**What to watch next:** (3-4 bullets)

---
## 3. Bull vs Bear
**Strongest Bull Case** (3-5 reasons + what must be true for it to play out)
**Strongest Bear Case** (3-5 reasons + what must be true for it to play out)
What would prove bulls wrong? What would prove bears wrong?
The 1-2 key variables that decide which side wins.
**What to watch next:** (3-4 bullets)

---
## 4. Multi-Timeframe Trend
- Short-term (days-weeks): based on RSI {rsi:.0f}, recent momentum {roc_1m:+.1f}%
- Medium-term (1-6 months): trend structure, mean reversion vs continuation
- Long-term (6-24 months): secular narrative, valuation sensitivity
How these trends may conflict and what a cautious retail trader should do.
**What to watch next:** (3-4 bullets)

---
## 5. Earnings Reality Check
What is likely already priced in? What would cause:
- Positive market reaction
- Negative reaction
- "Good numbers but stock down" outcome
- "Bad numbers but stock up" outcome
**What to watch next:** (3-4 bullets)

---
## 6. Sentiment vs Fundamentals
Compare what market sentiment seems to believe vs what fundamentals actually support.
Any disconnect? Squeeze risk? Rug-pull risk? Range-bound risk? Narrative rotation risk?
**What to watch next:** (3-4 bullets)

---
## 7. Scenario Planning (Next 6 Months)
**Bull Case:** Drivers, what improves, how to confirm
**Base Case:** Drivers, what stays stable, how to confirm
**Bear Case:** Drivers, what deteriorates, how to confirm
Single biggest swing factor.
Top 3 risks for a cautious retail trader.
Top 3 signals that conditions are improving or worsening.
**What to watch next:** (3-4 bullets)

---
## Executive Summary
- 3 bullish points
- 3 bearish points
- 3 watch items
- One sentence: "If I were a cautious retail investor in Indonesia, I would focus on ____."
"""

        resp = requests.post(
            CLAUDE_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        return f"Claude API error {resp.status_code}: {resp.text[:300]}"

    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}\n\nPlease check your API key or try again."


def parse_confidence(analysis_text):
    import re
    scores = {"buy": 33, "hold": 34, "sell": 33, "overall": "HOLD", "one_line": ""}
    try:
        if "CONFIDENCE_SCORES:" in analysis_text:
            block = analysis_text.split("CONFIDENCE_SCORES:")[1].split("END_SCORES")[0]
            for line in block.strip().splitlines():
                line = line.strip()
                m = re.search(r"\d+", line)
                if line.startswith("BUY:") and m:
                    scores["buy"] = int(m.group())
                elif line.startswith("HOLD:") and m:
                    scores["hold"] = int(m.group())
                elif line.startswith("SELL:") and m:
                    scores["sell"] = int(m.group())
                elif line.startswith("OVERALL:"):
                    scores["overall"] = line.split(":", 1)[1].strip()
                elif line.startswith("ONE_LINE:"):
                    scores["one_line"] = line.split(":", 1)[1].strip()
            clean = analysis_text.split("CONFIDENCE_SCORES:")[0]
            if "END_SCORES" in analysis_text:
                clean += analysis_text.split("END_SCORES")[1]
            return scores, clean.strip()
    except Exception:
        pass
    return scores, analysis_text

def make_chart(close, name, score, height=320):
    color = "#22c55e" if score >= 65 else "#f59e0b" if score >= 45 else "#ef4444"
    fig = go.Figure()
    sp = close.tail(252)
    fig.add_trace(go.Scatter(x=sp.index, y=sp.values, name=name,
                              line=dict(color=color, width=2)))
    for w, c, label in [(20,"#94a3b8","20DMA"),(50,"#f59e0b","50DMA"),(200,"#ef4444","200DMA")]:
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🇮🇩 Indo AI Dashboard")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    view = st.radio("View", [
        "🧠 AI Deep Analysis",
        "🔍 Quick Scores",
        "📊 Market Overview",
    ])
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M')}")
    st.caption("Powered by Google Gemini (free)")
    st.caption("⚠️ Not financial advice.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_us_macro_context():
    """Fetch key US macro signals to enrich AI analysis context."""
    try:
        spy   = fetch_series("SPY", "6mo")
        qqq   = fetch_series("QQQ", "6mo")
        vix   = fetch_series("^VIX", "1mo")
        dxy   = fetch_series("UUP",  "6mo")
        
        spy_1m  = roc(spy, 21)
        qqq_1m  = roc(qqq, 21)
        vix_now = latest(vix)
        dxy_1m  = roc(dxy, 21)
        spy_above_200 = latest(spy) > dma(spy, 200) if len(spy) >= 200 else None
        
        # Simple regime classification
        if not np.isnan(spy_1m) and not np.isnan(vix_now):
            if spy_1m > 3 and vix_now < 20:
                regime = "Risk-On"
            elif spy_1m < -5 or vix_now > 25:
                regime = "Risk-Off"
            else:
                regime = "Neutral/Transitioning"
        else:
            regime = "Unknown"
        
        return {
            "spy_1m":       round(spy_1m, 1) if not np.isnan(spy_1m) else None,
            "qqq_1m":       round(qqq_1m, 1) if not np.isnan(qqq_1m) else None,
            "vix":          round(vix_now, 1) if not np.isnan(vix_now) else None,
            "dxy_1m":       round(dxy_1m, 1) if not np.isnan(dxy_1m) else None,
            "spy_above_200":spy_above_200,
            "regime":       regime,
        }
    except:
        return {}

# ── Fetch shared macro data ───────────────────────────────────────────────────
idx_s  = fetch_series("^JKSE", "6mo")
idr_s  = fetch_series("IDR=X", "6mo")
em_s   = fetch_series("EEM",   "6mo")

idx_1m   = roc(idx_s, 21)
idr_1m   = roc(idr_s, 21)
em_1m    = roc(em_s,  21)
idx_now  = latest(idx_s)
idr_now  = latest(idr_s)

us_macro = fetch_us_macro_context()
us_macro_str = ""
if us_macro:
    us_macro_str = (f"SPY {us_macro.get('spy_1m',0):+.1f}% 1M, "
                   f"QQQ {us_macro.get('qqq_1m',0):+.1f}% 1M, "
                   f"VIX {us_macro.get('vix',20):.0f}, "
                   f"DXY {us_macro.get('dxy_1m',0):+.1f}% 1M, "
                   f"US Regime: {us_macro.get('regime','Unknown')}, "
                   f"SPY {'above' if us_macro.get('spy_above_200') else 'below'} 200DMA")

indo_macro_str = (f"IDX {idx_now:,.0f} ({idx_1m:+.1f}% 1M), "
                 f"USD/IDR {idr_now:,.0f} ({idr_1m:+.1f}% 1M), "
                 f"EM flows {em_1m:+.1f}% 1M")

macro_context = f"US macro: {us_macro_str} | Indonesia macro: {indo_macro_str}"

# ══════════════════════════════════════════════════════════════════════════════
# AI DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if view == "🧠 AI Deep Analysis":
    st.title("🧠 AI Deep Stock Analysis")
    st.caption("Institutional-grade analysis powered by Google Gemini")

    st.markdown("Enter any stock, crypto or ETF ticker:")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        custom_ticker = st.text_input("Ticker symbol", placeholder="e.g. BBCA.JK, AAPL, BTC-USD, NVDA, GOTO.JK").upper().strip()
    with col2:
        selected = st.selectbox("Or pick Indo stock", [""] + list(STOCKS.keys()))
    with col3:
        run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    # Determine which ticker to use
    if custom_ticker:
        ticker = custom_ticker
        # Auto-detect sector and name
        if any(x in ticker for x in ["BTC", "ETH", "XRP", "SOL", "DOGE", "BNB"]):
            sector = "Crypto"
            selected = ticker
        elif ticker.endswith(".JK"):
            sector = "Indonesian Stock"
            selected = ticker
        else:
            sector = "US Stock"
            selected = ticker
        stock_info = {"ticker": ticker, "sector": sector, "search": f"{ticker} stock news"}
    elif selected and selected in STOCKS:
        stock_info = STOCKS[selected]
        ticker = stock_info["ticker"]
        sector = stock_info["sector"]
    else:
        st.info("Enter a ticker symbol or select an Indo stock above, then click Run Full Analysis.")
        st.markdown("**Examples:**  (Apple),  (Nvidia),  (Microsoft),  (Bitcoin),  (BCA),  (GOTO)")
        st.stop()

    # Always show quick data
    with st.spinner("Loading market data..."):
        price_df = fetch_price(ticker, "1y")
        close    = get_close(price_df)
        volume   = get_volume(price_df)
        fund     = fetch_fundamentals(ticker)
        news     = fetch_news(stock_info["search"])

    p     = latest(close)
    r1d   = roc(close, 1)
    r1m   = roc(close, 21)
    r3m   = roc(close, 63)
    rsi   = compute_rsi(close)
    above_200 = (p > dma(close, 200)) if not np.isnan(dma(close, 200)) else False

    tech_score, tech_details = score_technical(close, volume)
    fund_score, fund_details = score_fundamentals(fund, sector)
    combined_score = tech_score * 0.4 + fund_score * 0.6
    em, sl, color = signal_color(combined_score)

    # Quick summary header
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: curr_sym, curr_code = get_currency(ticker)
    st.metric("Price", f"{curr_sym} {p:,.2f}" if not np.isnan(p) else "—",
                        delta=f"{r1d:+.2f}%" if not np.isnan(r1d) else None)
    with c2: st.metric("1 Month", f"{r1m:+.1f}%" if not np.isnan(r1m) else "—")
    with c3: st.metric("RSI", f"{rsi:.0f}" if not np.isnan(rsi) else "—")
    with c4: st.metric("Tech Score", f"{tech_score:.0f}%")
    with c5: st.metric("Fund Score", f"{fund_score:.0f}%")

    # Chart
    if not close.empty:
        st.plotly_chart(make_chart(close, selected, combined_score, 280), use_container_width=True)

    # Quick scores
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Technical Signals**")
        for d in tech_details[:5]:
            st.caption(d)
    with col2:
        st.markdown("**💰 Fundamental Signals**")
        for d in fund_details[:5]:
            st.caption(d)

    # Recent news preview
    if news:
        st.markdown("**📰 Recent News**")
        for item in news[:4]:
            st.caption(f"· [{item['title']}]({item['link']}) — *{item['age']}*")

    st.markdown("---")

    # AI Analysis
    if run_btn:
        pe_val    = fund.get("pe_ratio") or 0
        div_val   = (fund.get("dividend_yield") or 0) * 100
        eg_val    = (fund.get("earnings_growth") or 0) * 100
        rec_val   = fund.get("analyst_rec") or "N/A"
        rsi_val   = rsi if not np.isnan(rsi) else 50
        r1m_val   = r1m if not np.isnan(r1m) else 0
        r3m_val   = r3m if not np.isnan(r3m) else 0

        with st.spinner(f"🧠 Claude is analysing {selected}... this takes 15-30 seconds..."):
            analysis = run_gemini_analysis(
                stock_name=selected,
                ticker=ticker,
                sector=sector,
                price=p if not np.isnan(p) else 0,
                roc_1m=r1m_val,
                roc_3m=r3m_val,
                rsi=rsi_val,
                above_200dma=above_200,
                pe=pe_val,
                div_yield=div_val,
                earnings_growth=eg_val,
                analyst_rec=rec_val,
                news_headlines=news,
                macro_context=macro_context,
                tech_score=tech_score,
                fund_score=fund_score,
            )

        # Parse confidence scores
        conf, clean_analysis = parse_confidence(analysis)
        
        st.markdown("## 🧠 AI Analysis Report")
        st.markdown(f"*Generated by Claude (Anthropic) · {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        
        # Confidence score box
        overall = conf["overall"]
        buy_p = conf["buy"]
        hold_p = conf["hold"] 
        sell_p = conf["sell"]
        one_line = conf["one_line"]
        
        if overall == "BUY":
            box_color = "success"
        elif overall == "SELL":
            box_color = "error"
        else:
            box_color = "warning"
        
        getattr(st, box_color)(f"{'🟢' if overall=='BUY' else '🔴' if overall=='SELL' else '🟡'} **{overall}** — {buy_p if overall=='BUY' else hold_p if overall=='HOLD' else sell_p}% confidence")
        
        if one_line:
            st.caption(one_line)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🟢 BUY", f"{buy_p}%")
            st.progress(buy_p/100)
        with c2:
            st.metric("🟡 HOLD", f"{hold_p}%")
            st.progress(hold_p/100)
        with c3:
            st.metric("🔴 SELL", f"{sell_p}%")
            st.progress(sell_p/100)
        
        st.markdown("---")
        st.markdown(clean_analysis)
        st.markdown("---")
        st.caption("⚠️ This AI analysis is for informational purposes only. Not financial advice. Always do your own research before investing.")

    else:
        st.info("👆 Click **Run Full Analysis** to generate the AI report for this stock.")
        st.caption("Analysis takes 15-30 seconds. Uses Google Gemini free tier.")

# ══════════════════════════════════════════════════════════════════════════════
# QUICK SCORES
# ══════════════════════════════════════════════════════════════════════════════
elif view == "🔍 Quick Scores":
    st.title("🔍 Quick Stock Screener")

    sector_filter = st.selectbox("Filter by sector",
        ["All"] + sorted(list(set(v["sector"] for v in STOCKS.values()))))

    rows = []
    prog = st.progress(0)
    items = list(STOCKS.items())
    for i, (name, info) in enumerate(items):
        prog.progress((i+1)/len(items))
        if sector_filter != "All" and info["sector"] != sector_filter:
            continue
        close = fetch_series(info["ticker"])
        p     = latest(close)
        r1w   = roc(close, 5)
        r1m   = roc(close, 21)
        ts, _ = score_technical(close, pd.Series(dtype=float))
        fd    = fetch_fundamentals(info["ticker"])
        fs, _ = score_fundamentals(fd, info["sector"])
        overall = ts*0.4 + fs*0.6
        e2, sl2, _ = signal_color(overall)
        rows.append({
            "Stock":       name,
            "Sector":      info["sector"],
            "Price":       f"Rp {p:,.0f}" if not np.isnan(p) else "—",
            "1 Week":      f"{r1w:+.1f}%" if not np.isnan(r1w) else "—",
            "1 Month":     f"{r1m:+.1f}%" if not np.isnan(r1m) else "—",
            "Tech":        f"{ts:.0f}%",
            "Fundamentals":f"{fs:.0f}%",
            "Overall":     f"{overall:.0f}%",
            "Signal":      f"{e2} {sl2}",
            "_sort":       overall,
        })
    prog.empty()

    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("💡 Go to 'AI Deep Analysis' for the full 7-section Gemini report on any stock.")

# ══════════════════════════════════════════════════════════════════════════════
# MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif view == "📊 Market Overview":
    st.title("📊 Indonesia Market Overview")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("IDX Composite", f"{idx_now:,.0f}" if not np.isnan(idx_now) else "—",
                  delta=f"{idx_1m:+.1f}% (1mo)" if not np.isnan(idx_1m) else None)
    with c2:
        st.metric("USD/IDR", f"{idr_now:,.0f}" if not np.isnan(idr_now) else "—",
                  delta=f"{idr_1m:+.1f}% (1mo)" if not np.isnan(idr_1m) else None,
                  delta_color="inverse")
    with c3:
        em_now = latest(em_s)
        st.metric("EM ETF (EEM)", f"${em_now:.2f}" if not np.isnan(em_now) else "—",
                  delta=f"{em_1m:+.1f}% (1mo)" if not np.isnan(em_1m) else None)

    col1, col2 = st.columns(2)
    with col1:
        if not idx_s.empty:
            score = 65 if not np.isnan(idx_1m) and idx_1m > 0 else 35
            st.plotly_chart(make_chart(idx_s, "IDX", score, 250), use_container_width=True)
    with col2:
        if not idr_s.empty:
            score = 35 if not np.isnan(idr_1m) and idr_1m > 0 else 65
            st.plotly_chart(make_chart(idr_s, "USD/IDR", score, 250), use_container_width=True)

    st.markdown("### 📰 Latest Indonesia Market News")
    macro_news = fetch_news("Indonesia stock market economy 2025", 10)
    if macro_news:
        for item in macro_news:
            st.markdown(f"· [{item['title']}]({item['link']}) — *{item['age']}*")
    else:
        st.info("News feed loads on your local machine and Streamlit Cloud.")
