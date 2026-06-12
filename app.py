"""AI 股票预测 Pro · A 股行情分析、智能诊断、机器学习预测与策略回测。

运行方式：streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import advanced as adv_mod
from core import backtest as bt_mod
from core import data as data_mod
from core import indicators, news as news_mod
from core import model as model_mod
from core import screener as scr_mod
from core import signals as signals_mod
from core import watchlist as wl_mod

# 深度学习模型标签（torch 较重，core.deep 延迟到实际使用时再导入）
DEEP_CHOICES = {
    "LSTM (深度学习)": "lstm",
    "Transformer (深度学习)": "transformer",
}
ALL_MODEL_CHOICES = {**model_mod.MODEL_CHOICES, **DEEP_CHOICES}

st.set_page_config(page_title="AI 股票预测 Pro", page_icon="📈", layout="wide")

UP, DOWN = "#f23645", "#089981"  # A 股习惯：红涨绿跌
ACCENT, GOLD = "#6c8cff", "#f5b942"

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(108,140,255,0.08), rgba(108,140,255,0.02));
        border: 1px solid rgba(108,140,255,0.25);
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b97a7; }
    button[data-baseweb="tab"] { font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 数据与计算缓存 ----------

@st.cache_data(ttl=1800, show_spinner=False)
def load_history(symbol: str, years: int) -> pd.DataFrame:
    return data_mod.fetch_history(symbol, years=years)


@st.cache_data(ttl=86400, show_spinner=False)
def _load_name_cached(symbol: str) -> str:
    name = data_mod.fetch_stock_name(symbol)
    if name == symbol:
        # 抛异常以避免把「获取失败」缓存一整天，下次重试
        raise LookupError(symbol)
    return name


def load_name(symbol: str) -> str:
    try:
        return _load_name_cached(symbol)
    except Exception:
        return symbol


@st.cache_data(ttl=1800, show_spinner=False)
def run_forecast(df: pd.DataFrame, horizon: int, kind: str):
    if kind in DEEP_CHOICES.values():
        from core import deep

        return deep.train_and_forecast_deep(df, horizon=horizon, model_kind=kind)
    return model_mod.train_and_forecast(df, horizon=horizon, model_kind=kind)


@st.cache_data(ttl=1800, show_spinner=False)
def run_replay(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    # Stacking 单次训练成本高（内部含 12 次基模型拟合），降低重训频率
    retrain = 30 if kind == "stack" else 10
    return model_mod.walk_forward_replay(df, model_kind=kind, retrain_every=retrain)


@st.cache_data(ttl=1800, show_spinner=False)
def run_backtest(df: pd.DataFrame, kind: str, threshold: float, cost: float, sizing: str):
    retrain = 60 if kind == "stack" else 20
    return bt_mod.run_backtest(
        df, model_kind=kind, threshold=threshold, cost=cost, sizing=sizing, retrain_every=retrain
    )


@st.cache_data(ttl=1800, show_spinner=False)
def run_pro_eval(df: pd.DataFrame, kind: str):
    return adv_mod.professional_eval(df, model_kind=kind)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol: str) -> pd.DataFrame:
    return news_mod.fetch_news(symbol)


# ---------- 侧边栏 ----------

with st.sidebar:
    st.title("📈 AI 股票预测 Pro")
    symbol_input = st.text_input("股票代码", value="600519", help="6 位 A 股代码，如 600519（贵州茅台）、000001（平安银行）")
    years = st.select_slider("历史数据范围（年）", options=[1, 2, 3, 5, 8], value=3)
    horizon = st.slider("预测天数（交易日）", min_value=3, max_value=30, value=10)
    model_label = st.selectbox("预测模型", list(ALL_MODEL_CHOICES))
    run = st.button("开始分析", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ 本项目仅用于学习机器学习与量化分析技术，"
        "预测与回测结果不构成任何投资建议。股市有风险，入市需谨慎。"
    )

if not run and "df" not in st.session_state:
    st.info("👈 在左侧输入股票代码，点击「开始分析」")
    st.stop()

if run:
    try:
        with st.spinner("正在获取行情数据…"):
            symbol = data_mod.normalize_symbol(symbol_input)
            df = load_history(symbol, years)
            name = load_name(symbol)
        st.session_state.update(df=df, name=name, symbol=symbol, horizon=horizon, model_label=model_label)
        st.session_state.pop("bt_result", None)  # 切换股票后清除旧回测与对比
        st.session_state.pop("cmp_data", None)
    except Exception as e:
        st.error(f"数据获取失败：{e}")
        st.stop()

df: pd.DataFrame = st.session_state["df"]
name: str = st.session_state["name"]
symbol: str = st.session_state["symbol"]
horizon = st.session_state["horizon"]
model_label = st.session_state["model_label"]
model_kind = ALL_MODEL_CHOICES[model_label]
is_deep = model_kind in DEEP_CHOICES.values()

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

tab_k, tab_ind, tab_diag, tab_ai, tab_bt, tab_scr, tab_cmp, tab_wl, tab_news = st.tabs(
    ["🕯️ K 线与均线", "📊 技术指标", "🧭 智能诊断", "🤖 AI 预测", "⚖️ 策略回测", "🎯 智能选股", "🆚 多股对比", "⭐ 自选股监控", "📰 个股资讯"]
)

# ---------- K 线 ----------

with tab_k:
    show_cross = st.checkbox("标注 MACD 金叉 ▲ / 死叉 ▼", value=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=dfi["date"], open=dfi["open"], high=dfi["high"], low=dfi["low"], close=dfi["close"],
            name="K 线", increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
        ),
        row=1, col=1,
    )
    for w, color in [(5, GOLD), (20, "#42a5f5"), (60, "#ab47bc")]:
        fig.add_trace(
            go.Scatter(x=dfi["date"], y=dfi[f"ma{w}"], name=f"MA{w}", line=dict(width=1.2, color=color)),
            row=1, col=1,
        )
    if show_cross:
        golden, death = signals_mod.find_macd_crosses(dfi)
        fig.add_trace(
            go.Scatter(
                x=golden["date"], y=golden["low"] * 0.985, mode="markers", name="MACD 金叉",
                marker=dict(symbol="triangle-up", size=11, color=UP),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=death["date"], y=death["high"] * 1.015, mode="markers", name="MACD 死叉",
                marker=dict(symbol="triangle-down", size=11, color=DOWN),
            ),
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
        fig.add_scatter(x=dfi["date"], y=dfi["rsi14"], name="RSI14", line=dict(width=1.2, color=GOLD))
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

# ---------- 智能诊断 ----------

with tab_diag:
    diag = signals_mod.diagnose(df)

    col_score, col_items = st.columns([1, 1.6])
    with col_score:
        color = UP if diag.score >= 55 else (DOWN if diag.score <= 45 else GOLD)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=diag.score,
                number={"suffix": " 分"},
                title={"text": f"技术面综合评分 · <b>{diag.verdict}</b>"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(8,153,129,0.25)"},
                        {"range": [30, 45], "color": "rgba(8,153,129,0.12)"},
                        {"range": [45, 55], "color": "rgba(150,150,150,0.15)"},
                        {"range": [55, 70], "color": "rgba(242,54,69,0.12)"},
                        {"range": [70, 100], "color": "rgba(242,54,69,0.25)"},
                    ],
                },
            )
        )
        fig.update_layout(height=320, margin=dict(t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("评分基于最新交易日的 7 维技术信号加权汇总：50 为中性，越高越偏多。")

    with col_items:
        rows = []
        for item in diag.items:
            icon = {"多": "🔴 偏多", "空": "🟢 偏空", "中性": "⚪ 中性"}[item.verdict]
            rows.append({"指标": item.name, "当前状态": item.state, "信号": icon, "评分影响": f"{item.delta:+d}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=330)

    st.info("💡 信号颜色遵循 A 股习惯：红=偏多（看涨），绿=偏空（看跌）。评分为统计性参考，不构成投资建议。")

# ---------- AI 预测 ----------

with tab_ai:
    try:
        with st.spinner(f"正在训练 {model_label} 模型并预测…"):
            result = run_forecast(df, horizon, model_kind)
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
        name="AI 预测价", line=dict(color=GOLD, width=2, dash="dash"), mode="lines+markers",
    )
    fig.update_layout(height=440, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=30, b=10))
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
        st.download_button(
            "⬇️ 导出预测结果 CSV",
            show.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{symbol}_forecast.csv",
            mime="text/csv",
        )

    with col_i:
        if result.feature_importance is not None:
            st.markdown("**特征重要性 Top 10**")
            imp = result.feature_importance.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ACCENT))
            fig.update_layout(height=380, margin=dict(t=10, b=10, l=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("当前模型不提供特征重要性（融合/神经网络/线性模型）。")

    st.markdown("---")
    st.markdown("**🔁 模型可信度检验：历史预测 vs 实际走势（滚动回放）**")
    st.caption("模拟「每天只用当天之前的数据训练模型」，逐日预测次日收盘价并与实际对比——这是模型真实水平的试金石。")
    if is_deep:
        ex = result.extra or {}
        st.info(
            f"深度模型采用「时间留出验证」替代滚动回放（逐日重训成本过高）：以最近 {ex.get('val_size', '-')} 条序列为验证集，"
            f"上方方向准确率即验证集真实表现（训练 {ex.get('epochs', '-')} 轮，早停选优）。"
        )
    else:
        try:
            with st.spinner("正在滚动回放历史预测…"):
                replay = run_replay(df, model_kind)
            hit = float((replay["pred_ret"] * replay["actual_ret"] > 0).mean())
            r1, r2 = st.columns([3, 1])
            with r1:
                fig = go.Figure()
                fig.add_scatter(x=replay["date"], y=replay["actual"], name="实际收盘价", line=dict(color="#42a5f5", width=1.6))
                fig.add_scatter(x=replay["date"], y=replay["predicted"], name="当日模型预测", line=dict(color=GOLD, width=1.4, dash="dot"))
                fig.update_layout(height=360, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with r2:
                st.metric("回放区间方向命中率", f"{hit * 100:.1f}%")
                st.metric("回放天数", f"{len(replay)}")
                mae_replay = float((replay["pred_ret"] - replay["actual_ret"]).abs().mean())
                st.metric("收益率 MAE", f"{mae_replay * 100:.2f}%")
        except Exception as e:
            st.warning(f"滚动回放失败：{e}")

    st.markdown("---")
    st.markdown("**🎓 专业信号评估：Purged 交叉验证 · IC / RankIC / 分位数分析 · 方向概率**")
    eval_kind = "lgbm" if is_deep else model_kind
    eval_note = "（深度模型的信号评估采用 LightGBM 代理）" if is_deep else ""
    st.caption(
        f"采用专业量化机构的标准评估体系{eval_note}：训练/测试集之间留出 5 日 embargo 空窗防泄漏；"
        "IC > 0.03 即认为信号有效，分位数收益单调递增说明信号区分度强。"
    )
    try:
        with st.spinner("正在进行 Purged 交叉验证评估…"):
            pe = run_pro_eval(df, eval_kind)

        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("IC（信息系数）", f"{pe.ic:.4f}", "有效 ≥ 0.03" if pe.ic >= 0.03 else "偏弱 < 0.03", delta_color="off")
        e2.metric("RankIC", f"{pe.rank_ic:.4f}")
        e3.metric("ICIR（稳定性）", f"{pe.icir:.2f}")
        e4.metric("方向 AUC", f"{pe.direction_auc:.3f}", "随机=0.500", delta_color="off")
        prob_color = "🔴" if pe.prob_up >= 0.55 else ("🟢" if pe.prob_up <= 0.45 else "⚪")
        e5.metric(f"{prob_color} 次日上涨概率", f"{pe.prob_up * 100:.1f}%")

        q1, q2 = st.columns([1.2, 1])
        with q1:
            qt = pe.quantile_table
            colors = [UP if v >= 0 else DOWN for v in qt["平均实际收益"]]
            fig = go.Figure(
                go.Bar(
                    x=qt["分位"], y=qt["平均实际收益"] * 100,
                    marker_color=colors,
                    text=[f"{v * 100:+.3f}%" for v in qt["平均实际收益"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                height=320, margin=dict(t=30, b=10),
                yaxis_title="实际平均次日收益 %", title="预测分位 vs 实际收益（应单调递增）",
            )
            st.plotly_chart(fig, use_container_width=True)
        with q2:
            show_qt = qt.copy()
            show_qt["平均实际收益"] = (show_qt["平均实际收益"] * 100).map("{:+.3f}%".format)
            show_qt["胜率"] = (show_qt["胜率"] * 100).map("{:.1f}%".format)
            st.dataframe(show_qt, use_container_width=True, hide_index=True)
            st.caption(
                f"基于 {pe.n_samples} 个 out-of-fold 样本。Q5（最看多组）收益显著高于 Q1（最看空组）"
                "说明模型确实捕捉到了有效信号，而非随机噪声。"
            )
    except Exception as e:
        st.warning(f"专业评估暂不可用：{e}")

    st.warning(
        "📢 **免责声明**：以上预测基于历史技术面数据的统计规律，无法预知突发消息、政策与基本面变化。"
        "股价短期走势接近随机游走，任何模型的预测都存在很大不确定性。"
        "本工具仅供学习参考，切勿作为实际投资依据。"
    )

# ---------- 策略回测 ----------

with tab_bt:
    st.markdown("**模型信号策略**：滚动训练（不使用未来数据），预测次日收益为正则持仓、否则空仓，与「买入持有」对比。")
    if is_deep:
        st.info("策略回测需要逐段重训模型，深度模型成本过高，此处自动改用「岭回归」信号回测。")
    bt_kind = "ridge" if is_deep else model_kind
    p1, p2, p3, p4 = st.columns(4)
    threshold = p1.number_input("开仓阈值（预测次日收益 >）", value=0.0, step=0.001, format="%.3f")
    cost = p2.number_input("单边交易成本", value=0.001, step=0.0005, format="%.4f", help="含佣金与冲击成本，0.001 = 0.1%")
    sizing_label = p3.selectbox(
        "仓位方式", ["二元（满仓/空仓）", "置信度加权（0~100%）"],
        help="置信度加权：仓位与预测强度成正比，模型把握越大仓位越高，能显著降低回撤",
    )
    sizing = "confidence" if sizing_label.startswith("置信度") else "binary"
    p4.markdown(" ")
    do_bt = p4.button("🚀 运行回测（最近一年）", use_container_width=True)

    if do_bt:
        try:
            with st.spinner("正在滚动训练并回测（约 10-60 秒）…"):
                st.session_state["bt_result"] = run_backtest(df, bt_kind, threshold, cost, sizing)
        except Exception as e:
            st.error(f"回测失败：{e}")

    bt = st.session_state.get("bt_result")
    if bt is not None:
        m = bt.metrics
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("策略总收益", f"{m['策略总收益'] * 100:+.1f}%", f"持有 {m['买入持有总收益'] * 100:+.1f}%", delta_color="off")
        b2.metric("策略年化", f"{m['策略年化收益'] * 100:+.1f}%", f"持有 {m['买入持有年化'] * 100:+.1f}%", delta_color="off")
        b3.metric("最大回撤", f"{m['策略最大回撤'] * 100:.1f}%", f"持有 {m['买入持有最大回撤'] * 100:.1f}%", delta_color="off")
        b4.metric("夏普比率", f"{m['夏普比率']:.2f}")
        b5.metric("持仓日胜率", f"{m['持仓日胜率'] * 100:.1f}%", f"换仓 {bt.n_trades} 次", delta_color="off")

        eq = bt.equity
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
        fig.add_scatter(x=eq["date"], y=(eq["strategy"] - 1) * 100, name="模型信号策略", line=dict(color=GOLD, width=2), row=1, col=1)
        fig.add_scatter(x=eq["date"], y=(eq["buy_hold"] - 1) * 100, name="买入持有", line=dict(color="#42a5f5", width=1.6), row=1, col=1)
        fig.add_scatter(
            x=eq["date"], y=eq["position"], name="仓位（0~1）",
            line=dict(color=ACCENT, width=1), fill="tozeroy", fillcolor="rgba(108,140,255,0.18)", row=2, col=1,
        )
        fig.update_yaxes(title_text="累计收益 %", row=1, col=1)
        fig.update_yaxes(range=[-0.05, 1.1], row=2, col=1)
        fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("回测期为最近约 250 个交易日；每 20 个交易日用截至当日的数据重新训练模型，全程无未来数据泄漏。")
    else:
        st.info("设置参数后点击「运行回测」查看策略表现。")

# ---------- 智能选股 ----------

with tab_scr:
    st.markdown(
        "**两阶段漏斗选股**：全市场约 5000 只股票 → 快照粗筛（剔除 ST/停牌/小市值/低流动性，"
        "按 60 日动量与资金活跃度保留候选池）→ **横截面 LightGBM 精排**"
        "（用全部候选股约 1.5 年的 40 维因子面板训练「未来 5 日收益」模型，"
        "样本量是单股模型的数百倍）→ 综合 AI 预测收益、动量、技术诊断评分输出最终排名。"
    )

    s1, s2, s3, s4 = st.columns(4)
    pool_size = s1.selectbox("候选池规模（粗筛保留）", [200, 300, 500], index=1)
    top_k = s2.selectbox("最终选出数量", [50, 100], index=1)
    min_cap = s3.number_input("最小总市值（亿元）", value=30, step=10, min_value=10)
    boards = s4.multiselect("排除板块", ["创业板", "科创板", "北交所"], default=["北交所"])

    do_screen = st.button("🎯 开始全市场选股（约 1-3 分钟）", type="primary")

    if do_screen:
        prog = st.progress(0.0, text="正在获取全市场快照（约 10-30 秒）…")
        try:
            spot = scr_mod.fetch_market_snapshot()
            prog.progress(0.08, text=f"快照完成（{len(spot)} 只），粗筛中…")
            cand = scr_mod.coarse_screen(
                spot, pool_size=pool_size, min_cap_yi=float(min_cap), exclude_boards=tuple(boards)
            )

            def _cb(done, total, stage):
                if stage == "拉取行情":
                    frac = 0.08 + 0.52 * done / total
                elif stage == "构造因子":
                    frac = 0.60 + 0.30 * done / total
                elif stage == "训练横截面模型":
                    frac = 0.92
                else:
                    frac = 1.0
                prog.progress(min(frac, 1.0), text=f"{stage}（{done}/{total}）…")

            res = scr_mod.fine_rank(cand, top_k=top_k, progress=_cb)
            prog.empty()
            st.session_state["scr_result"] = (res, len(spot))
        except Exception as e:
            prog.empty()
            st.error(f"选股失败：{e}")

    scr_state = st.session_state.get("scr_result")
    if scr_state:
        res, n_spot = scr_state
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("全市场扫描", f"{n_spot} 只")
        g2.metric("粗筛候选", f"{res.n_candidates} 只")
        g3.metric("完成精排", f"{res.n_ok} 只", f"失败 {len(res.failed)}", delta_color="off")
        g4.metric("横截面训练样本", f"{res.train_samples:,}")

        show = res.table.copy()
        for col in ("20日涨跌", "60日涨跌", "AI预测5日收益"):
            show[col] = (show[col] * 100).map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "-")
        show["综合得分"] = show["综合得分"].map("{:.1f}".format)
        show["最新价"] = show["最新价"].map("{:.2f}".format)
        st.dataframe(show, use_container_width=True, hide_index=True, height=600)

        st.download_button(
            "⬇️ 导出选股结果 CSV",
            show.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"top{len(show)}_picks.csv",
            mime="text/csv",
        )
        st.caption(
            "综合得分 = 50% AI 预测收益排名 + 25% 动量排名 + 25% 技术诊断分。"
            "可将感兴趣的代码输入左侧进行单股深度分析，或加入自选股监控。"
        )

    st.warning(
        "📢 **重要提示**：选股基于历史量价统计规律，市场存在突发消息、政策等模型无法预知的因素，"
        "任何选股系统都无法保证未来上涨。结果仅供学习参考，切勿直接作为投资依据。"
    )

# ---------- 多股对比 ----------

with tab_cmp:
    cmp_input = st.text_input("输入多只股票代码（逗号分隔）", value=f"{symbol}, 000001, 300750")
    if st.button("📊 开始对比"):
        codes = []
        for c in cmp_input.replace("，", ",").split(","):
            c = c.strip()
            if c:
                try:
                    codes.append(data_mod.normalize_symbol(c))
                except ValueError:
                    st.warning(f"已忽略无效代码：{c}")
        codes = list(dict.fromkeys(codes))[:6]
        if len(codes) < 2:
            st.error("请至少输入 2 个有效代码")
        else:
            series, names, failed = {}, {}, []
            with st.spinner("正在获取各股票行情…"):
                for c in codes:
                    try:
                        d = load_history(c, years)
                        series[c] = d.set_index("date")["close"]
                        names[c] = load_name(c)
                    except Exception:
                        failed.append(c)
            if failed:
                st.warning(f"以下代码获取失败：{', '.join(failed)}")
            if len(series) >= 2:
                st.session_state["cmp_data"] = (series, names)

    cmp_state = st.session_state.get("cmp_data")
    if cmp_state:
        series, names = cmp_state
        aligned = pd.DataFrame(series).dropna()
        norm = aligned / aligned.iloc[0] * 100

        fig = go.Figure()
        for c in norm.columns:
            fig.add_scatter(x=norm.index, y=norm[c], name=f"{names[c]}({c})", line=dict(width=1.6))
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
        fig.update_layout(
            height=420, hovermode="x unified", legend=dict(orientation="h"),
            margin=dict(t=30, b=10), yaxis_title="归一化价格（起点=100）",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**区间表现**")
            rows = []
            for c in aligned.columns:
                s = aligned[c]
                ret = s.iloc[-1] / s.iloc[0] - 1
                dd = float((s / s.cummax() - 1).min())
                vol = float(s.pct_change().std() * (252 ** 0.5))
                rows.append({"股票": f"{names[c]}({c})", "区间涨跌": f"{ret * 100:+.1f}%", "年化波动": f"{vol * 100:.1f}%", "最大回撤": f"{dd * 100:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**日收益相关性**")
            corr = aligned.pct_change().corr()
            labels = [f"{names[c]}({c})" for c in corr.columns]
            fig = go.Figure(
                go.Heatmap(
                    z=corr.values, x=labels, y=labels,
                    colorscale="RdBu", zmin=-1, zmax=1, text=corr.round(2).values,
                    texttemplate="%{text}",
                )
            )
            # 标签可能是纯数字字符串，强制按分类轴处理，避免被当作数值坐标
            fig.update_xaxes(type="category")
            fig.update_yaxes(type="category")
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ---------- 自选股监控 ----------

with tab_wl:
    codes = wl_mod.load_watchlist()

    col_add, col_del, col_run = st.columns([1.2, 1.2, 1])
    with col_add:
        new_code = st.text_input("添加自选股", placeholder="输入 6 位代码，如 601318", key="wl_add")
        if st.button("➕ 添加") and new_code.strip():
            try:
                codes = wl_mod.add_stock(new_code)
                st.success(f"已添加 {wl_mod.data_mod.normalize_symbol(new_code)}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    with col_del:
        del_code = st.selectbox("移除自选股", ["（选择代码）", *codes], key="wl_del")
        if st.button("🗑️ 移除") and del_code in codes:
            wl_mod.remove_stock(del_code)
            st.rerun()
    with col_run:
        st.markdown(" ")
        st.markdown(" ")
        do_inspect = st.button("🔍 一键巡检", type="primary", use_container_width=True)

    st.caption(f"当前自选股（{len(codes)} 只）：{'、'.join(codes)}")

    if do_inspect:
        with st.spinner(f"正在巡检 {len(codes)} 只股票（每只约 5-10 秒）…"):
            st.session_state["wl_report"] = wl_mod.inspect_all(codes)

    report = st.session_state.get("wl_report")
    if report is not None:
        ok = report[report["诊断评分"].notna()]
        s1, s2, s3 = st.columns(3)
        s1.metric("🔴 AI 看多", int((ok["AI信号"] == "看多").sum()))
        s2.metric("🟢 AI 看空", int((ok["AI信号"] == "看空").sum()))
        s3.metric("平均诊断评分", f"{ok['诊断评分'].mean():.0f} 分" if not ok.empty else "-")

        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 导出巡检报告 CSV",
            report.to_csv(index=False).encode("utf-8-sig"),
            file_name="watchlist_report.csv",
            mime="text/csv",
        )
        st.caption("AI 次日预测来自岭回归快速模型（horizon=1）；详细分析请在左侧输入单只代码查看。")

    with st.expander("⏰ 设置每日定时自动预测（Windows 计划任务）"):
        st.markdown(
            "项目自带 `daily_report.py`，每天收盘后自动巡检自选股并把报告保存到 `reports/` 目录。\n\n"
            "在 PowerShell（管理员）中运行以下命令，即可注册每个交易日 15:30 自动执行：\n"
        )
        import sys as _sys

        task_cmd = (
            f'schtasks /Create /TN "AI股票每日预测" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:30 '
            f'/TR "cmd /c cd /d \\"{wl_mod.os.path.dirname(wl_mod.os.path.abspath(wl_mod.__file__))}\\\\..\\" '
            f'&& \\"{_sys.executable}\\" daily_report.py"'
        )
        st.code(task_cmd, language="powershell")
        st.markdown("取消任务：`schtasks /Delete /TN \"AI股票每日预测\" /F`")

# ---------- 个股资讯 ----------

with tab_news:
    try:
        with st.spinner("正在获取最新资讯…"):
            news_df = load_news(symbol)
        senti_counts = news_df["sentiment"].value_counts()
        n1, n2, n3 = st.columns(3)
        n1.metric("🔴 偏利好", int(senti_counts.get("偏利好", 0)))
        n2.metric("⚪ 中性", int(senti_counts.get("中性", 0)))
        n3.metric("🟢 偏利空", int(senti_counts.get("偏利空", 0)))

        icon_map = {"偏利好": "🔴", "中性": "⚪", "偏利空": "🟢"}
        for _, row in news_df.iterrows():
            icon = icon_map.get(row["sentiment"], "⚪")
            st.markdown(
                f"{icon} **[{row['title']}]({row['url']})**  \n"
                f"<span style='color:#8b97a7;font-size:13px'>{row['time']} · {row['source']} · {row['sentiment']}</span>",
                unsafe_allow_html=True,
            )
        st.caption("情绪标注基于标题/正文关键词的简单规则，仅供参考。")
    except Exception as e:
        st.info(f"资讯获取暂不可用（{type(e).__name__}），不影响其他功能。")
