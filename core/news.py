"""个股资讯获取（东方财富搜索接口）与轻量级情绪标注（关键词规则，无额外依赖）。

akshare 的 stock_news_em 封装参数已过时，这里直接调用东财搜索 API。
"""

from __future__ import annotations

import json
import re

import pandas as pd
import requests

from . import data as _data

_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"

_POSITIVE = (
    "上涨", "大涨", "涨停", "增长", "利好", "突破", "新高", "回购", "增持",
    "超预期", "盈利", "扭亏", "中标", "签约", "合作", "获批", "创新高", "买入",
)
_NEGATIVE = (
    "下跌", "大跌", "跌停", "下滑", "利空", "亏损", "新低", "减持", "退市",
    "违规", "处罚", "诉讼", "风险", "下调", "低于预期", "质押", "卖出", "警示",
)

_TAG_RE = re.compile(r"</?em>|\(|\)")


def _sentiment(text: str) -> str:
    pos = sum(text.count(w) for w in _POSITIVE)
    neg = sum(text.count(w) for w in _NEGATIVE)
    if pos > neg:
        return "偏利好"
    if neg > pos:
        return "偏利空"
    return "中性"


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text or "").replace("\u3000", "").strip()


def fetch_news(symbol: str, limit: int = 20) -> pd.DataFrame:
    """获取个股最新新闻并标注情绪倾向。失败时抛出异常，由界面层兜底。"""
    symbol = _data.normalize_symbol(symbol)
    param = {
        "uid": "",
        "keyword": symbol,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": limit,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    r = requests.get(
        _NEWS_URL,
        params={"cb": "cb", "param": json.dumps(param, ensure_ascii=False)},
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    r.raise_for_status()
    body = r.text.strip()
    data = json.loads(body[body.index("(") + 1 : body.rindex(")")])
    articles = (data.get("result") or {}).get("cmsArticleWebOld") or []
    if not articles:
        raise ValueError("暂无相关资讯")

    rows = []
    for a in articles[:limit]:
        title = _clean(a.get("title", ""))
        content = _clean(a.get("content", ""))
        rows.append(
            {
                "time": a.get("date", ""),
                "title": title,
                "sentiment": _sentiment(title + content),
                "source": a.get("mediaName", ""),
                "url": a.get("url", ""),
            }
        )
    return pd.DataFrame(rows)
