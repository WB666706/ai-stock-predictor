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


def fetch_history(
    symbol: str, years: float = 3, adjust: str = "qfq", prefer_tencent: bool = False
) -> pd.DataFrame:
    """获取指定股票最近 N 年的日线历史行情（默认前复权）。

    默认优先东方财富源（字段更全），连接失败时自动切换到腾讯源；
    批量场景（如全市场选股）可设 prefer_tencent=True，腾讯接口更快更稳。
    """
    symbol = normalize_symbol(symbol)
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365.25))
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    sources = [("东方财富", _fetch_eastmoney), ("腾讯", _fetch_tencent)]
    if prefer_tencent:
        sources.reverse()

    df, errors = None, []
    for name, fetcher in sources:
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


def _index_df(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return (
        df.dropna()
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_index_history(symbol: str = "000001", years: int = 9) -> pd.DataFrame:
    """获取大盘指数日线收盘价（默认上证指数），用于构造市场相对强度 / Beta 特征。

    直连东方财富 K 线接口（akshare 的指数接口需先拉全量指数列表，慢且易断连），
    失败时切换腾讯备用源。
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365.25))

    for attempt in range(2):
        try:
            r = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={
                    "secid": f"1.{symbol}",
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f53",  # 日期、收盘
                    "klt": "101",
                    "fqt": "1",
                    "beg": start.strftime("%Y%m%d"),
                    "end": "20500101",
                },
                timeout=10,
            )
            klines = (r.json().get("data") or {}).get("klines") or []
            if klines:
                return _index_df([tuple(k.split(",")) for k in klines])
        except (requests.exceptions.RequestException, ConnectionError, ValueError):
            time.sleep(1)

    # 腾讯接口单次最多约 640 条，按 880 个自然日（约 600 个交易日）分窗拉取
    rows: list[tuple] = []
    cur = start
    while cur <= end:
        win_end = min(cur + dt.timedelta(days=880), end)
        r = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"sh{symbol},day,{cur:%Y-%m-%d},{win_end:%Y-%m-%d},640,qfq"},
            timeout=10,
        )
        payload = (r.json().get("data") or {}).get(f"sh{symbol}") or {}
        klines = payload.get("qfqday") or payload.get("day") or []
        rows.extend((k[0], k[2]) for k in klines)
        cur = win_end + dt.timedelta(days=1)
    return _index_df(rows)


_SPOT_FIELDS = {
    "f12": "代码",
    "f14": "名称",
    "f2": "最新价",
    "f3": "涨跌幅",
    "f6": "成交额",
    "f8": "换手率",
    "f10": "量比",
    "f20": "总市值",
    "f24": "60日涨跌幅",
}


def fetch_spot_fast(max_workers: int = 8) -> pd.DataFrame:
    """全市场 A 股实时快照（直连东财 clist 接口，并行分页，约 10 秒）。

    返回列：代码、名称、最新价、涨跌幅、成交额、换手率、量比、总市值、60日涨跌幅。
    """
    from concurrent.futures import ThreadPoolExecutor

    base_params = {
        "po": "1", "np": "1", "fltt": "2", "invt": "2", "fid": "f3",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": ",".join(_SPOT_FIELDS),
        "pz": "100",
    }

    def _page(pn: int) -> list[dict]:
        r = requests.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={**base_params, "pn": str(pn)},
            timeout=15,
        )
        return ((r.json().get("data") or {}).get("diff")) or []

    first = requests.get(
        "https://push2.eastmoney.com/api/qt/clist/get",
        params={**base_params, "pn": "1"},
        timeout=15,
    ).json()
    total = (first.get("data") or {}).get("total") or 0
    if total < 1000:
        raise ConnectionError("东财快照接口返回异常")
    rows = list((first["data"].get("diff")) or [])

    n_pages = -(-total // 100)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for page_rows in pool.map(_page, range(2, n_pages + 1)):
            rows.extend(page_rows)

    df = pd.DataFrame(rows).rename(columns=_SPOT_FIELDS)
    for c in df.columns:
        if c not in ("代码", "名称"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates(subset="代码").reset_index(drop=True)


def fetch_history_fast(symbol: str, years: float = 1.5) -> pd.DataFrame:
    """单只股票日线快速获取（直连腾讯 fqkline，单请求约 0.3-1 秒，最多 640 根）。

    返回列：date / open / close / high / low / volume（前复权），适合批量选股场景。
    """
    symbol = normalize_symbol(symbol)
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365.25))
    full = f"{market_prefix(symbol)}{symbol}"

    r = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{full},day,{start:%Y-%m-%d},{end:%Y-%m-%d},640,qfq"},
        timeout=10,
    )
    payload = (r.json().get("data") or {}).get(full) or {}
    klines = payload.get("qfqday") or payload.get("day") or []
    if not klines:
        raise ValueError(f"未获取到 {symbol} 的行情数据")

    df = pd.DataFrame(
        [k[:6] for k in klines],
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


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
