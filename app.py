"""AI 股票预测 · A 股行情分析与机器学习走势预测（Streamlit 应用）。

运行方式：streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import data as data_mod
from core import indicators, model as model_mod

st.set_page_config(page_title="AI 股票预测", page_icon="📈", layout="wide")

UP, DOWN = "#f23645", "#089981"  # A 股习惯：红涨绿跌


# ---------- 数据缓存 ----------

@st.cache_data(ttl=1800, show_spinner=False)
def load_history(symbol: str, years: int) -> pd.DataFrame:
    return data_mod.fetch_history(symbol, years=years)


@st.cache_data(ttl=86400, show_spinner=False)
def load_name(symbol: str) -> str:
    return data_mod.fetch_stock_name(symbol)


# ---------- 侧边栏 ----------

with st.sidebar:
    st.title("📈 AI 股票预测")
    symbol_input = st.text_input("股票代码", value="600519", help="6 位 A 股代码，如 600519（贵州茅台）、000001（平安银行）")
    years = st.select_slider("历史数据范围（年）", options=[1, 2, 3, 5, 8], value=3)
    horizon = st.slider("预测天数（交易日）", min_value=3, max_value=30, value=10)
    model_label = st.selectbox("预测模型", list(model_mod.MODEL_CHOICES))
    run = st.button("开始分析", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ 本项目仅用于学习机器学习与量化分析技术，"
        "预测结果不构成任何投资建议。股市有风险，入市需谨慎。"
    )

if not run and "df" not in st.session_state:
    st.info("👈 在左侧输入股票代码，点击「开始分析」")
    st.stop()

# ---------- 加载数据 ----------

if run:
    try:
        with st.spinner("正在获取行情数据…"):
            symbol = data_mod.normalize_symbol(symbol_input)
            df = load_history(symbol, years)
            name = load_name(symbol)
        st.session_state.update(df=df, name=name, symbol=symbol, horizon=horizon, model_label=model_label)
    except Exception as e:
        st.error(f"数据获取失败：{e}")
        st.stop()

df: pd.DataFrame = st.session_state["df"]
name: str = st.session_state["name"]
symbol: str = st.session_state["symbol"]
horizon = st.session_state["horizon"]
model_label = st.session_state["model_label"]

dfi = indicators.add_all(df)

# ---------- 概览指标 ----------

last, prev = df.iloc[-1], df.iloc[-2]
pct = (last["close"] / prev["close"] - 1) * 100

st.subheader(f"{name}（{symbol}）")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新收盘", f"{last['close']:.2f}", f"{pct:+.2f}%")
c2.metric("最高 / 最低", f"{last['high']:.2f} / {last['low']:.2f}")
c3.metric("成交量（手）", f"{last['volume'] / 1e4:.1f} 万")
c4.metric("区间最高", f"{df['high'].max():.2f}")
c5.metric("区间最低", f"{df['low'].min():.2f}")
st.caption(f"数据范围：{df['date'].iloc[0]:%Y-%m-%d} ～ {df['date'].iloc[-1]:%Y-%m-%d}，前复权日线，共 {len(df)} 个交易日")

tab_k, tab_ind, tab_ai = st.tabs(["🕯️ K 线与均线", "📊 技术指标", "🤖 AI 预测"])

# ---------- K 线 ----------

with tab_k:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=dfi["date"], open=dfi["open"], high=dfi["high"], low=dfi["low"], close=dfi["close"],
            name="K 线", increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
        ),
        row=1, col=1,
    )
    for w, color in [(5, "#f5b942"), (20, "#42a5f5"), (60, "#ab47bc")]:
        fig.add_trace(
            go.Scatter(x=dfi["date"], y=dfi[f"ma{w}"], name=f"MA{w}", line=dict(width=1.2, color=color)),
            row=1, col=1,
        )
    vol_colors = [UP if c >= o else DOWN for c, o in zip(dfi["close"], dfi["open"])]
    fig.add_trace(
        go.Bar(x=dfi["date"], y=dfi["volume"], name="成交量", marker_color=vol_colors, opacity=0.7),
        row=2, col=1,
    )
    fig.update_layout(
        height=620, xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- 技术指标 ----------

with tab_ind:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**MACD**")
        fig = go.Figure()
        hist_colors = [UP if v >= 0 else DOWN for v in dfi["macd_hist"].fillna(0)]
        fig.add_bar(x=dfi["date"], y=dfi["macd_hist"], name="MACD 柱", marker_color=hist_colors)
        fig.add_scatter(x=dfi["date"], y=dfi["macd_dif"], name="DIF", line=dict(width=1.2))
        fig.add_scatter(x=dfi["date"], y=dfi["macd_dea"], name="DEA", line=dict(width=1.2))
        fig.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**KDJ**")
        fig = go.Figure()
        for col_name, label in [("kdj_k", "K"), ("kdj_d", "D"), ("kdj_j", "J")]:
            fig.add_scatter(x=dfi["date"], y=dfi[col_name], name=label, line=dict(width=1.2))
        fig.add_hline(y=80, line_dash="dot", line_color="gray")
        fig.add_hline(y=20, line_dash="dot", line_color="gray")
        fig.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**RSI（14 日）**")
        fig = go.Figure()
        fig.add_scatter(x=dfi["date"], y=dfi["rsi14"], name="RSI14", line=dict(width=1.2, color="#f5b942"))
        fig.add_hline(y=70, line_dash="dot", line_color=UP, annotation_text="超买 70")
        fig.add_hline(y=30, line_dash="dot", line_color=DOWN, annotation_text="超卖 30")
        fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**布林带（BOLL 20, 2σ）**")
        fig = go.Figure()
        fig.add_scatter(x=dfi["date"], y=dfi["boll_upper"], name="上轨", line=dict(width=1, color="rgba(242,54,69,0.6)"))
        fig.add_scatter(
            x=dfi["date"], y=dfi["boll_lower"], name="下轨",
            line=dict(width=1, color="rgba(8,153,129,0.6)"), fill="tonexty", fillcolor="rgba(120,120,200,0.08)",
        )
        fig.add_scatter(x=dfi["date"], y=dfi["boll_mid"], name="中轨", line=dict(width=1, dash="dot"))
        fig.add_scatter(x=dfi["date"], y=dfi["close"], name="收盘价", line=dict(width=1.4, color="#42a5f5"))
        fig.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

# ---------- AI 预测 ----------

with tab_ai:
    model_kind = model_mod.MODEL_CHOICES[model_label]
    try:
        with st.spinner(f"正在训练 {model_label} 模型并预测…"):
            result = model_mod.train_and_forecast(df, horizon=horizon, model_kind=model_kind)
    except Exception as e:
        st.error(f"模型训练失败：{e}")
        st.stop()

    fc = result.forecast
    final = fc.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"{horizon} 个交易日后预测价",
        f"{final['price']:.2f}",
        f"{final['cum_return'] * 100:+.2f}%",
    )
    m2.metric(
        "次日方向准确率（回测）",
        f"{result.direction_accuracy * 100:.1f}%",
        f"基准 {result.baseline_accuracy * 100:.1f}%",
        delta_color="off",
    )
    m3.metric("次日收益率 MAE", f"{result.mae * 100:.2f}%")
    m4.metric("训练样本数", f"{result.n_samples}")

    # 历史价格（最近 120 日）+ 预测曲线与置信区间
    hist = df.tail(120)
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["close"], name="历史收盘价", line=dict(color="#42a5f5", width=1.6))

    bridge_x = [hist["date"].iloc[-1], *fc["date"]]
    fig.add_scatter(
        x=bridge_x, y=[hist["close"].iloc[-1], *fc["upper"]],
        name="95% 区间上界", line=dict(width=0), showlegend=False,
    )
    fig.add_scatter(
        x=bridge_x, y=[hist["close"].iloc[-1], *fc["lower"]],
        name="95% 置信区间", line=dict(width=0), fill="tonexty", fillcolor="rgba(245,185,66,0.18)",
    )
    fig.add_scatter(
        x=bridge_x, y=[hist["close"].iloc[-1], *fc["price"]],
        name="AI 预测价", line=dict(color="#f5b942", width=2, dash="dash"), mode="lines+markers",
    )
    fig.update_layout(height=460, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col_t, col_i = st.columns([1.2, 1])
    with col_t:
        st.markdown("**预测明细**")
        show = fc.copy()
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        show["cum_return"] = (show["cum_return"] * 100).map("{:+.2f}%".format)
        show = show[["date", "price", "lower", "upper", "cum_return"]].rename(
            columns={"date": "日期", "price": "预测价", "lower": "下界", "upper": "上界", "cum_return": "累计涨跌幅"}
        )
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)

    with col_i:
        if result.feature_importance is not None:
            st.markdown("**特征重要性 Top 10**")
            imp = result.feature_importance.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#6c8cff"))
            fig.update_layout(height=380, margin=dict(t=10, b=10, l=10))
            st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "📢 **免责声明**：以上预测基于历史技术面数据的统计规律，无法预知突发消息、政策与基本面变化。"
        "股价短期走势接近随机游走，任何模型的预测都存在很大不确定性。"
        "本工具仅供学习参考，切勿作为实际投资依据。"
    )
