"""智能技术面诊断：汇总多维度技术信号，生成 0-100 综合评分与买卖点标注。

评分仅基于技术面统计规律，不构成投资建议。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators


@dataclass
class SignalItem:
    name: str  # 指标名称
    state: str  # 当前状态描述
    verdict: str  # 多 / 空 / 中性
    delta: int  # 对综合评分的加减分


@dataclass
class Diagnosis:
    score: int  # 0-100 综合评分
    verdict: str  # 总体结论
    items: list[SignalItem]


def find_macd_crosses(dfi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (金叉点, 死叉点)，用于在 K 线图上标注。"""
    dif, dea = dfi["macd_dif"], dfi["macd_dea"]
    golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    death = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    return dfi.loc[golden.fillna(False)], dfi.loc[death.fillna(False)]


def _recent_cross(dfi: pd.DataFrame, days: int = 5) -> str:
    """检查最近 N 日内是否出现 MACD 金叉/死叉。"""
    golden, death = find_macd_crosses(dfi.tail(days + 1))
    if not golden.empty:
        return "golden"
    if not death.empty:
        return "death"
    return "none"


def diagnose(df: pd.DataFrame) -> Diagnosis:
    """对最新交易日做技术面综合诊断。"""
    dfi = indicators.add_all(df)
    last = dfi.iloc[-1]
    items: list[SignalItem] = []

    def add(name: str, state: str, verdict: str, delta: int):
        items.append(SignalItem(name, state, verdict, delta))

    # 1. 均线排列
    close, ma5, ma20, ma60 = last["close"], last["ma5"], last["ma20"], last["ma60"]
    if close > ma5 > ma20 > ma60:
        add("均线系统", "多头排列（价 > MA5 > MA20 > MA60）", "多", 15)
    elif close < ma5 < ma20 < ma60:
        add("均线系统", "空头排列（价 < MA5 < MA20 < MA60）", "空", -15)
    elif close > ma20:
        add("均线系统", "价格站上 MA20，中期偏强", "多", 6)
    else:
        add("均线系统", "价格跌破 MA20，中期偏弱", "空", -6)

    # 2. MACD
    cross = _recent_cross(dfi)
    if cross == "golden":
        add("MACD", "近 5 日出现金叉", "多", 12)
    elif cross == "death":
        add("MACD", "近 5 日出现死叉", "空", -12)
    elif last["macd_dif"] > last["macd_dea"]:
        add("MACD", "DIF 在 DEA 上方运行", "多", 6)
    else:
        add("MACD", "DIF 在 DEA 下方运行", "空", -6)

    # 3. RSI
    rsi = last["rsi14"]
    if rsi >= 70:
        add("RSI(14)", f"{rsi:.0f}，进入超买区，回调风险增大", "空", -8)
    elif rsi <= 30:
        add("RSI(14)", f"{rsi:.0f}，进入超卖区，存在反弹机会", "多", 8)
    elif rsi >= 50:
        add("RSI(14)", f"{rsi:.0f}，多方力量占优", "多", 4)
    else:
        add("RSI(14)", f"{rsi:.0f}，空方力量占优", "空", -4)

    # 4. KDJ
    k, d, j = last["kdj_k"], last["kdj_d"], last["kdj_j"]
    if j > 100:
        add("KDJ", f"J={j:.0f}，严重超买", "空", -6)
    elif j < 0:
        add("KDJ", f"J={j:.0f}，严重超卖", "多", 6)
    elif k > d:
        add("KDJ", f"K({k:.0f}) 在 D({d:.0f}) 上方，短线偏强", "多", 5)
    else:
        add("KDJ", f"K({k:.0f}) 在 D({d:.0f}) 下方，短线偏弱", "空", -5)

    # 5. 布林带位置
    width = last["boll_upper"] - last["boll_lower"]
    pos = (close - last["boll_lower"]) / width if width > 0 else 0.5
    if pos >= 0.95:
        add("布林带", "触及上轨，短期涨幅过大", "空", -5)
    elif pos <= 0.05:
        add("布林带", "触及下轨，超跌状态", "多", 5)
    elif pos >= 0.5:
        add("布林带", f"运行于中轨上方（位置 {pos:.0%}）", "多", 3)
    else:
        add("布林带", f"运行于中轨下方（位置 {pos:.0%}）", "空", -3)

    # 6. 量能
    vol_ma5 = dfi["volume"].tail(5).mean()
    vol_ratio = last["volume"] / vol_ma5 if vol_ma5 > 0 else 1
    day_up = last["close"] >= last["open"]
    if vol_ratio >= 1.5 and day_up:
        add("量能", f"放量上涨（量比 {vol_ratio:.1f}），资金活跃", "多", 6)
    elif vol_ratio >= 1.5 and not day_up:
        add("量能", f"放量下跌（量比 {vol_ratio:.1f}），抛压明显", "空", -6)
    elif vol_ratio <= 0.6:
        add("量能", f"明显缩量（量比 {vol_ratio:.1f}），观望情绪浓", "中性", 0)
    else:
        add("量能", f"量能正常（量比 {vol_ratio:.1f}）", "中性", 0)

    # 7. 中期动量
    if len(dfi) > 20:
        mom20 = close / dfi["close"].iloc[-21] - 1
        if mom20 >= 0.10:
            add("20 日动量", f"近月上涨 {mom20:+.1%}，趋势强劲", "多", 5)
        elif mom20 <= -0.10:
            add("20 日动量", f"近月下跌 {mom20:+.1%}，趋势疲弱", "空", -5)
        else:
            add("20 日动量", f"近月涨跌 {mom20:+.1%}，趋势平缓", "中性", 0)

    score = max(0, min(100, 50 + sum(i.delta for i in items)))
    if score >= 70:
        verdict = "强势看多"
    elif score >= 55:
        verdict = "偏多"
    elif score > 45:
        verdict = "中性"
    elif score > 30:
        verdict = "偏空"
    else:
        verdict = "弱势看空"

    return Diagnosis(score=score, verdict=verdict, items=items)
