"""
Indonesian Stock + Crypto Dashboard
Top 45 IHSG stocks + BTC, ETH, XRP
Run with: python -m streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Indo Market Dashboard",
    page_icon="🇮🇩",
    layout="wide",
)

# ── Tickers ───────────────────────────────────────────────────────────────────

TOP45_IHSG = {
    "BCA":            "BBCA.JK",
    "BRI":            "BBRI.JK",
    "Telkom":         "TLKM.JK",
    "Mandiri":        "BMRI.JK",
    "Astra":          "ASII.JK",
    "BNI":            "BBNI.JK",
    "Chandra Asri":   "TPIA.JK",
    "Bayan Coal":     "BYAN.JK",
    "Adaro":          "ADRO.JK",
    "Merdeka Copper": "MDKA.JK",
    "Indofood CB":    "ICBP.JK",
    "Indofood":       "INDF.JK",
    "Unilever":       "UNVR.JK",
    "Kalbe Farma":    "KLBF.JK",
    "Gudang Garam":   "GGRM.JK",
    "HM Sampoerna":   "HMSP.JK",
    "Semen Indonesia":"SMGR.JK",
    "Vale Indonesia": "INCO.JK",
    "Aneka Tambang":  "ANTM.JK",
    "Bukit Asam":     "PTBA.JK",
    "GOTO":           "GOTO.JK",
    "Bank BTN":       "BBTN.JK",
    "Bank CIMB":      "BNGA.JK",
    "Bank Danamon":   "BDMN.JK",
    "Bank Panin":     "PNBN.JK",
    "Maybank Indo":   "BNII.JK",
    "Medco Energy":   "MEDC.JK",
    "Elnusa":         "ELSA.JK",
    "Indika Energy":  "INDY.JK",
    "Harum Energy":   "HRUM.JK",
    "Mitra Keluarga": "MIKA.JK",
    "Siloam Hosp":    "SILO.JK",
    "Pakuwon":        "PWON.JK",
    "Summarecon":     "SMRA.JK",
    "Ciputra":        "CTRA.JK",
    "Bumi Serpong":   "BSDE.JK",
    "Jasa Marga":     "JSMR.JK",
    "Wijaya Karya":   "WIKA.JK",
    "Waskita Karya":  "WSKT.JK",
    "PP Tbk":         "PTPP.JK",
    "Ace Hardware":   "ACES.JK",
    "Matahari Dept":  "LPPF.JK",
    "Ramayana":       "RALS.JK",
    "XL Axiata":      "EXCL.JK",
    "Indosat":        "ISAT.JK",
}

CRYPTO = {
    "Bitcoin (BTC)":  "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "XRP (Ripple)":   "XRP-USD",
}

IDX_TICKER    = "^JKSE"
USDIDR_TICKER = "IDR=X"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker: str, period: str = "6mo") -> pd.Series:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        if isinstance(df.columns, pd.MultiIndex):
            s = df["Close"].iloc[:, 0]
        else:
            s = df["Close"]
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, name=ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all():
    out = {}
    all_tickers = {**TOP45_IHSG, **CRYPTO}
    for name, ticker in all_tickers.items():
        out[name] = fetch(ticker)
    return out

def dma(s, w):
    if len(s) < w:
        return float("nan")
    return float(s.rolling(w).mean().iloc[-1])

def roc(s, d):
    if len(s) < d + 1:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-d - 1] - 1) * 100)

def latest(s):
    return float(s.iloc[-1]) if not s.empty else float("nan")

def stock_score(s):
    if s.empty or len(s) < 10:
        return float("nan")
    score = 50.0
    p = latest(s)
    if not np.isnan(dma(s, 50)): score += 15 if p > dma(s, 50) else -15
    if not np.isnan(dma(s, 20)): score += 10 if p > dma(s, 20) else -10
    r1m = roc(s, 21)
    r1w = roc(s, 5)
    if not np.isnan(r1m): score += min(max(r1m * 1.5, -20), 20)
    if not np.isnan(r1w): score += min(max(r1w * 2, -10), 10)
    return max(0, min(100, score))

def signal_label(score):
    if np.isnan(score): return "—", "—"
    if score >= 65: return "🟢", "BUY / ADD"
    if score >= 45: return "🟡", "HOLD"
    return "🔴", "WAIT"

def make_chart(series, name, color="#3b82f6", show_50=True, show_200=False, height=280):
    fig = go.Figure()
    sp = series.tail(180)
    fig.add_trace(go.Scatter(x=sp.index, y=sp.values, name=name,
                              line=dict(color=color, width=2), showlegend=True))
    if show_50 and len(series) >= 50:
        ma50 = series.rolling(50).mean().tail(180)
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name="50DMA",
                                  line=dict(color="#f59e0b", width=1, dash="dot"), showlegend=True))
    if show_200 and len(series) >= 200:
        ma200 = series.rolling(200).mean().tail(180)
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="200DMA",
                                  line=dict(color="#ef4444", width=1, dash="dot"), showlegend=True))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), showlegend=True,
                      legend=dict(orientation="h", y=1.1),
                      xaxis=dict(gridcolor="#1f2937"),
                      yaxis=dict(gridcolor="#1f2937"))
    return fig

# ── Fetch data ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🇮🇩 Indo Dashboard")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    view = st.radio("View", ["📊 Overview", "🏦 All Stocks", "₿ Crypto", "📈 All Charts"])
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M')}")
    st.caption("⚠️ Not financial advice.")
    with st.expander("ℹ️ How signals work"):
        st.markdown("""
**Score 0–100:**
- 🟢 65+ = Good time to consider buying
- 🟡 45–64 = Hold, wait for clarity
- 🔴 0–44 = Risky, avoid adding

**Looks at:**
- Price vs 20 & 50 day averages
- 1 week & 1 month performance

**Remember:** This is a guide, not a guarantee. Always do your own research.
        """)

with st.spinner("Loading data for 45 stocks + crypto..."):
    idx_data  = fetch(IDX_TICKER)
    idr_data  = fetch(USDIDR_TICKER)
    all_data  = fetch_all()

ihsg_data   = {k: all_data[k] for k in TOP45_IHSG}
crypto_data = {k: all_data[k] for k in CRYPTO}

# ── Market signal ─────────────────────────────────────────────────────────────
def market_signal():
    score = 50.0
    notes = []
    idx_now = latest(idx_data)
    idx_50  = dma(idx_data, 50)
    idx_1m  = roc(idx_data, 21)
    if not np.isnan(idx_now) and not np.isnan(idx_50):
        if idx_now > idx_50:
            score += 15
            notes.append("✅ Jakarta market trending UP (IDX above 50-day average)")
        else:
            score -= 15
            notes.append("⚠️ Jakarta market trending DOWN (IDX below 50-day average)")
    if not np.isnan(idx_1m):
        if idx_1m > 2:
            score += 10
            notes.append(f"✅ IDX up {idx_1m:.1f}% this month — positive momentum")
        elif idx_1m < -2:
            score -= 10
            notes.append(f"⚠️ IDX down {idx_1m:.1f}% this month — losing momentum")
        else:
            notes.append(f"➡️ IDX roughly flat this month ({idx_1m:+.1f}%)")
    idr_1m = roc(idr_data, 21)
    if not np.isnan(idr_1m):
        if idr_1m > 1.5:
            score -= 15
            notes.append(f"⚠️ Rupiah weakening {idr_1m:+.1f}% vs USD — foreign investors may sell")
        elif idr_1m < -1.5:
            score += 15
            notes.append(f"✅ Rupiah strengthening {idr_1m:+.1f}% vs USD — positive for stocks")
        else:
            notes.append(f"➡️ Rupiah stable vs USD ({idr_1m:+.1f}%)")
    above = sum(1 for s in ihsg_data.values() if not s.empty and len(s) >= 50 and latest(s) > dma(s, 50))
    total = sum(1 for s in ihsg_data.values() if not s.empty and len(s) >= 50)
    if total > 0:
        pct = above / total
        if pct >= 0.6:
            score += 10
            notes.append(f"✅ {above}/{total} stocks above 50-day average — broad market strength")
        elif pct <= 0.35:
            score -= 10
            notes.append(f"⚠️ Only {above}/{total} stocks above 50-day average — market is weak")
        else:
            notes.append(f"➡️ {above}/{total} stocks above 50-day average — mixed market")
    return max(0, min(100, score)), notes

mkt_score, mkt_notes = market_signal()
em, sl = signal_label(mkt_score)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view == "📊 Overview":
    st.title("📊 Market Overview")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"## {em} Overall Signal: **{sl}**")
        st.progress(int(mkt_score) / 100)
        st.metric("Market Health Score", f"{mkt_score:.0f} / 100")
        if mkt_score >= 65:
            st.success("Conditions look good. Market is healthy and trending up.")
        elif mkt_score >= 45:
            st.warning("Mixed signals. Don't make big moves — wait for clarity.")
        else:
            st.error("Conditions unfavorable. Be careful adding new positions.")

    st.markdown("---")
    st.markdown("### What the market is saying")
    for n in mkt_notes:
        st.markdown(n)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        idx_now = latest(idx_data)
        idx_1m  = roc(idx_data, 21)
        st.metric("IDX Composite (Jakarta)", f"{idx_now:,.0f}" if not np.isnan(idx_now) else "—",
                  delta=f"{idx_1m:+.1f}% (1 month)" if not np.isnan(idx_1m) else None)
        if not idx_data.empty:
            st.plotly_chart(make_chart(idx_data, "IDX", "#3b82f6", height=240), use_container_width=True)

    with col2:
        idr_now = latest(idr_data)
        idr_1m  = roc(idr_data, 21)
        st.metric("USD/IDR (lower = stronger Rupiah ✅)",
                  f"{idr_now:,.0f}" if not np.isnan(idr_now) else "—",
                  delta=f"{idr_1m:+.1f}% (1 month)" if not np.isnan(idr_1m) else None,
                  delta_color="inverse")
        if not idr_data.empty:
            st.plotly_chart(make_chart(idr_data, "USD/IDR", "#ef4444", show_50=False, height=240),
                            use_container_width=True)

    st.markdown("---")
    st.markdown("### ₿ Crypto Today")
    cc = st.columns(3)
    for i, (name, series) in enumerate(crypto_data.items()):
        with cc[i]:
            p = latest(series)
            r1d = roc(series, 1)
            r1m = roc(series, 21)
            sc = stock_score(series)
            e2, sl2 = signal_label(sc)
            st.metric(name, f"${p:,.2f}" if not np.isnan(p) else "—",
                      delta=f"{r1d:+.2f}% today" if not np.isnan(r1d) else None)
            st.caption(f"1 month: {r1m:+.1f}%" if not np.isnan(r1m) else "")
            st.caption(f"{e2} {sl2} (score: {sc:.0f})" if not np.isnan(sc) else "")

# ══════════════════════════════════════════════════════════════════════════════
# ALL STOCKS
# ══════════════════════════════════════════════════════════════════════════════
elif view == "🏦 All Stocks":
    st.title("🏦 Top 45 IHSG Stocks")

    rows = []
    for name, series in ihsg_data.items():
        p   = latest(series)
        r1d = roc(series, 1)
        r1w = roc(series, 5)
        r1m = roc(series, 21)
        d50 = dma(series, 50)
        sc  = stock_score(series)
        vs50 = ((p / d50) - 1) * 100 if not (np.isnan(p) or np.isnan(d50)) else float("nan")
        e2, sl2 = signal_label(sc)
        rows.append({
            "Stock":    name,
            "Price":    f"Rp {p:,.0f}" if not np.isnan(p) else "—",
            "Today":    f"{r1d:+.1f}%" if not np.isnan(r1d) else "—",
            "1 Week":   f"{r1w:+.1f}%" if not np.isnan(r1w) else "—",
            "1 Month":  f"{r1m:+.1f}%" if not np.isnan(r1m) else "—",
            "vs 50DMA": f"{vs50:+.1f}%" if not np.isnan(vs50) else "—",
            "Signal":   f"{e2} {sl2}",
            "_score":   sc if not np.isnan(sc) else 0,
        })

    df = pd.DataFrame(rows).sort_values("_score", ascending=False).drop(columns=["_score"])

    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search", "")
    with col2:
        sig_f = st.selectbox("Filter", ["All signals", "🟢 BUY only", "🟡 HOLD only", "🔴 WAIT only"])

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["Stock"].str.contains(search, case=False)]
    if "BUY" in sig_f:
        filtered = filtered[filtered["Signal"].str.contains("BUY")]
    elif "HOLD" in sig_f:
        filtered = filtered[filtered["Signal"].str.contains("HOLD")]
    elif "WAIT" in sig_f:
        filtered = filtered[filtered["Signal"].str.contains("WAIT")]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Individual Stock Chart")
    selected = st.selectbox("Choose a stock", list(TOP45_IHSG.keys()))
    series = ihsg_data[selected]
    if not series.empty:
        sc = stock_score(series)
        e2, sl2 = signal_label(sc)
        r1m = roc(series, 21)
        st.metric(selected, f"Rp {latest(series):,.0f}",
                  delta=f"{r1m:+.1f}% (1 month)" if not np.isnan(r1m) else None)
        color = "#22c55e" if sc >= 65 else "#f59e0b" if sc >= 45 else "#ef4444"
        fig = make_chart(series, selected, color=color, height=320)
        fig.update_layout(title=f"{selected} — {e2} {sl2} (Score: {sc:.0f})")
        st.plotly_chart(fig, use_container_width=True)
        if sc >= 65:
            st.success(f"{e2} {sl2}: This stock is above its averages and trending positively.")
        elif sc >= 45:
            st.warning(f"{e2} {sl2}: Mixed signals. Don't rush in — watch for a few more days.")
        else:
            st.error(f"{e2} {sl2}: Stock is weak. Better to wait before buying more.")
    else:
        st.warning(f"No data available for {selected}")

# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO
# ══════════════════════════════════════════════════════════════════════════════
elif view == "₿ Crypto":
    st.title("₿ Crypto — BTC · ETH · XRP")

    for name, series in crypto_data.items():
        st.markdown(f"### {name}")
        p   = latest(series)
        r1d = roc(series, 1)
        r1w = roc(series, 5)
        r1m = roc(series, 21)
        sc  = stock_score(series)
        e2, sl2 = signal_label(sc)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Price", f"${p:,.2f}" if not np.isnan(p) else "—",
                            delta=f"{r1d:+.2f}% today" if not np.isnan(r1d) else None)
        with c2: st.metric("1 Week", f"{r1w:+.1f}%" if not np.isnan(r1w) else "—")
        with c3: st.metric("1 Month", f"{r1m:+.1f}%" if not np.isnan(r1m) else "—")
        with c4: st.metric("Signal", f"{e2} {sl2}")

        if not series.empty:
            color = "#22c55e" if sc >= 65 else "#f59e0b" if sc >= 45 else "#ef4444"
            st.plotly_chart(
                make_chart(series, name, color=color, show_50=True, show_200=True, height=300),
                use_container_width=True
            )

        if sc >= 65:
            st.success(f"Crypto momentum is positive. Above key averages.")
        elif sc >= 45:
            st.warning(f"Mixed signals. Watch before adding.")
        else:
            st.error(f"Weak momentum. Be cautious.")
        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ALL CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif view == "📈 All Charts":
    st.title("📈 All 45 Stock Charts")
    st.caption("Green = BUY · Orange = HOLD · Red = WAIT")

    items = list(ihsg_data.items())
    for i in range(0, len(items), 3):
        cols = st.columns(3)
        for j, (name, series) in enumerate(items[i:i+3]):
            with cols[j]:
                if series.empty:
                    st.caption(f"{name}: no data")
                    continue
                sc = stock_score(series)
                e2, sl2 = signal_label(sc)
                color = "#22c55e" if sc >= 65 else "#f59e0b" if sc >= 45 else "#ef4444"
                fig = go.Figure()
                sp = series.tail(126)
                fig.add_trace(go.Scatter(x=sp.index, y=sp.values,
                                          line=dict(color=color, width=1.5), showlegend=False))
                if len(series) >= 50:
                    ma50 = series.rolling(50).mean().tail(126)
                    fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values,
                                              line=dict(color="#94a3b8", width=1, dash="dot"),
                                              showlegend=False))
                fig.update_layout(
                    title=dict(text=f"{name}  {e2} {sc:.0f}", font=dict(size=11, color=color)),
                    height=175, margin=dict(l=5, r=5, t=28, b=5),
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font=dict(color="#fafafa", size=9),
                    xaxis=dict(gridcolor="#1f2937", showticklabels=False),
                    yaxis=dict(gridcolor="#1f2937"),
                )
                st.plotly_chart(fig, use_container_width=True)
