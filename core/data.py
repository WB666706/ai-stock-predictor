"""A 股历史行情数据获取（基于 akshare，东方财富为主、腾讯为备用数据源）。"""

from __future__ import annotations

import datetime as dt
import os
import time

# 行情接口都是国内数据源，走代理（科学上网工具）反而常常连不上，这里显式绕过
_no_proxy = {
    "eastmoney.com",
    "gtimg.cn",
    *filter(None, os.environ.get("NO_PROXY", "").split(",")),
}
os.environ["NO_PROXY"] = ",".join(_no_proxy)
os.environ["no_proxy"] = os.environ["NO_PROXY"]

import requests

# 东方财富接口会间歇性拒绝无浏览器 User-Agent 的请求（RemoteDisconnected），
# akshare 内部直接调用 requests.get，这里统一补上默认 UA
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
_orig_get = requests.get


def _get_with_ua(*args, **kwargs):
    kwargs.setdefault("headers", {}).setdefault("User-Agent", _UA)
    return _orig_get(*args, **kwargs)


requests.get = _get_with_ua

import akshare as ak
import pandas as pd

# akshare 返回的中文列名 -> 英文标准列名
_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}


def normalize_symbol(raw: str) -> str:
    """清洗用户输入的股票代码，返回 6 位数字代码。"""
    code = raw.strip().lower()
    for prefix in ("sh", "sz", "bj"):
        code = code.removeprefix(prefix)
    code = code.replace(".", "").strip()
    if not (code.isdigit() and len(code) == 6):
        raise ValueError(f"无效的股票代码：{raw!r}，请输入 6 位数字代码，如 600519")
    return code


def market_prefix(symbol: str) -> str:
    """根据 6 位代码推断交易所前缀：sh / sz / bj。"""
    if symbol[0] in ("6", "9", "5"):
        return "sh"
    if symbol[0] in ("4", "8"):
        return "bj"
    return "sz"


def _fetch_eastmoney(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust
    )
    return df.rename(columns=_COLUMN_MAP)


def _fetch_tencent(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist_tx(
        symbol=f"{market_prefix(symbol)}{symbol}", start_date=start, end_date=end, adjust=adjust
    )
    # 腾讯源的 amount 列实为成交量（手）
    return df.rename(columns={"amount": "volume"})


def fetch_history(symbol: str, years: int = 3, adjust: str = "qfq") -> pd.DataFrame:
    """获取指定股票最近 N 年的日线历史行情（默认前复权）。

    优先使用东方财富源（字段更全），连接失败时自动切换到腾讯源。
    """
    symbol = normalize_symbol(symbol)
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365.25))
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    df, errors = None, []
    for name, fetcher in (("东方财富", _fetch_eastmoney), ("腾讯", _fetch_tencent)):
        for attempt in range(2):
            try:
                df = fetcher(symbol, start_s, end_s, adjust)
                break
            except (requests.exceptions.RequestException, ConnectionError) as e:
                errors.append(f"{name}: {type(e).__name__}")
                time.sleep(1)
        if df is not None:
            break
    if df is None:
        raise ConnectionError(f"所有行情数据源均连接失败（{'; '.join(errors)}），请稍后重试")
    if df.empty:
        raise ValueError(f"未获取到股票 {symbol} 的行情数据，请检查代码是否正确")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def fetch_stock_name(symbol: str) -> str:
    """获取股票名称，东财失败时改用腾讯行情接口，再失败返回代码本身。"""
    symbol = normalize_symbol(symbol)
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        row = info.loc[info["item"] == "股票简称", "value"]
        if not row.empty:
            return str(row.iloc[0])
    except Exception:
        pass
    try:
        r = requests.get(
            f"https://qt.gtimg.cn/q={market_prefix(symbol)}{symbol}", timeout=5
        )
        r.encoding = "gbk"
        # 返回格式：v_sh600519="1~贵州茅台~600519~..."
        fields = r.text.split("~")
        if len(fields) > 2 and fields[1].strip():
            return fields[1].strip()
    except Exception:
        pass
    return symbol
