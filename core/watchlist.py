"""自选股管理与一键巡检。

自选股列表持久化到项目根目录 watchlist.json；
巡检时对每只股票汇总：最新价、涨跌幅、技术面诊断评分、AI 次日信号。
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import data as data_mod
from . import model as model_mod
from . import signals as signals_mod

_WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "..", "watchlist.json")
DEFAULT_WATCHLIST = ["600519", "000001", "300750"]


def load_watchlist() -> list[str]:
    try:
        with open(_WATCHLIST_PATH, encoding="utf-8") as f:
            codes = json.load(f)
        if isinstance(codes, list) and codes:
            return [str(c) for c in codes]
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(codes: list[str]) -> None:
    with open(_WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(codes, f, ensure_ascii=False, indent=2)


def add_stock(code: str) -> list[str]:
    code = data_mod.normalize_symbol(code)
    codes = load_watchlist()
    if code not in codes:
        codes.append(code)
        save_watchlist(codes)
    return codes


def remove_stock(code: str) -> list[str]:
    codes = [c for c in load_watchlist() if c != code]
    save_watchlist(codes)
    return codes


def inspect_stock(code: str, years: int = 2, model_kind: str = "ridge") -> dict:
    """巡检单只股票，返回概览 + 诊断 + AI 次日信号。"""
    df = data_mod.fetch_history(code, years=years)
    name = data_mod.fetch_stock_name(code)
    last, prev = df.iloc[-1], df.iloc[-2]

    diag = signals_mod.diagnose(df)

    result = model_mod.train_and_forecast(df, horizon=1, model_kind=model_kind)
    next_ret = float(result.forecast.iloc[0]["cum_return"])

    return {
        "代码": code,
        "名称": name,
        "最新价": round(float(last["close"]), 2),
        "今日涨跌": f"{(last['close'] / prev['close'] - 1) * 100:+.2f}%",
        "5日涨跌": f"{(last['close'] / df['close'].iloc[-6] - 1) * 100:+.2f}%" if len(df) > 5 else "-",
        "诊断评分": diag.score,
        "诊断结论": diag.verdict,
        "AI次日预测": f"{next_ret * 100:+.2f}%",
        "AI信号": "看多" if next_ret > 0 else "看空",
    }


def inspect_all(codes: list[str] | None = None, model_kind: str = "ridge") -> pd.DataFrame:
    """巡检自选股列表，单只失败不影响整体。"""
    codes = codes or load_watchlist()
    rows = []
    for code in codes:
        try:
            rows.append(inspect_stock(code, model_kind=model_kind))
        except Exception as e:
            rows.append(
                {
                    "代码": code, "名称": "获取失败", "最新价": None, "今日涨跌": "-",
                    "5日涨跌": "-", "诊断评分": None, "诊断结论": f"{type(e).__name__}",
                    "AI次日预测": "-", "AI信号": "-",
                }
            )
    return pd.DataFrame(rows)
