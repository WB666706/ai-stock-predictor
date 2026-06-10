"""常用技术指标计算：MA / MACD / RSI / BOLL / KDJ。

所有函数接收带有 open/high/low/close/volume 列的 DataFrame，
返回追加了指标列的副本，不修改原数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_ma(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 60)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_dif"] = ema_fast - ema_slow
    df["macd_dea"] = df["macd_dif"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi{window}"] = 100 - 100 / (1 + rs)
    return df


def add_boll(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["boll_mid"] = mid
    df["boll_upper"] = mid + num_std * std
    df["boll_lower"] = mid - num_std * std
    return df


def add_kdj(df: pd.DataFrame, window: int = 9) -> pd.DataFrame:
    df = df.copy()
    low_n = df["low"].rolling(window).min()
    high_n = df["high"].rolling(window).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """一次性追加全部指标。"""
    df = add_ma(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_boll(df)
    df = add_kdj(df)
    return df
