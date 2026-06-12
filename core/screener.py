"""智能选股引擎：两阶段漏斗从全市场约 5000 只股票中选出最可能上涨的 Top N。

阶段一 · 粗筛（秒级，基于全市场实时快照）：
- 剔除 ST/退市风险、停牌、低价股、小市值、流动性不足的股票
- 按 60 日动量、量比、换手率活跃度等快照因子打分，保留前 N 只候选

阶段二 · 精排（约 1-2 分钟，基于候选股约 1.5 年日线）：
- 对每只候选股构造 40 维量价因子，拼成「横截面面板」
- 训练横截面 LightGBM 模型（目标：未来 5 日收益）——量化机构的标准做法：
  用同一时期所有股票的数据学习「什么样的形态接下来会涨」，
  样本量是单股模型的数百倍，泛化能力显著更强
- 综合 AI 预测收益、动量、技术面诊断评分，输出最终 Top N

防泄漏说明：面板训练样本的「未来 5 日收益」目标在最近 5 个交易日内不存在
（shift 后为 NaN 被剔除），模型预测的最新截面不参与训练。

⚠️ 选股结果为统计性参考，不构成投资建议。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as data_mod
from . import model as model_mod
from . import signals as signals_mod

# 各板块代码前缀
BOARD_PREFIX = {
    "创业板": ("300", "301"),
    "科创板": ("688", "689"),
    "北交所": ("4", "8", "9"),
}


def fetch_market_snapshot() -> pd.DataFrame:
    """获取全市场 A 股实时快照（约 5000 行），含价格、动量、量比、市值等。

    优先直连东财 clist 接口（并行分页，约 10 秒）；失败时回退 akshare（较慢）。
    """
    try:
        return data_mod.fetch_spot_fast()
    except Exception:
        pass

    import akshare as ak

    last_err: Exception | None = None
    for _ in range(2):
        try:
            spot = ak.stock_zh_a_spot_em()
            if spot is not None and len(spot) > 1000:
                return spot
        except Exception as e:  # noqa: BLE001 - 接口偶发断连，统一重试
            last_err = e
            time.sleep(1.5)
    raise ConnectionError(f"全市场快照获取失败：{last_err}")


def _rank_pct(s: pd.Series) -> pd.Series:
    """百分位排名（0~1），NaN 记 0.5（中性）。"""
    return s.rank(pct=True).fillna(0.5)


def coarse_screen(
    spot: pd.DataFrame,
    pool_size: int = 300,
    min_cap_yi: float = 30.0,
    min_price: float = 2.0,
    exclude_st: bool = True,
    exclude_boards: tuple = ("北交所",),
) -> pd.DataFrame:
    """阶段一：快照粗筛，返回按粗筛得分排序的候选池 DataFrame。"""
    df = spot.copy()
    num_cols = ["最新价", "量比", "换手率", "60日涨跌幅", "涨跌幅", "总市值", "成交额"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = (
        df["最新价"].notna()
        & (df["最新价"] >= min_price)
        & (df["总市值"] >= min_cap_yi * 1e8)
        & (df["成交额"] > 1e7)          # 当日成交额 > 1000 万，剔除僵尸股
        & df["量比"].notna()            # 量比为空通常为停牌
    )
    if exclude_st:
        mask &= ~df["名称"].str.contains("ST|退", na=False)
    for board in exclude_boards:
        prefixes = BOARD_PREFIX.get(board, ())
        if prefixes:
            mask &= ~df["代码"].str.startswith(prefixes)
    df = df[mask].copy()

    # 粗筛得分：中期动量为主，叠加资金活跃度；换手率取「适度活跃」带通
    turnover_score = 1 - (df["换手率"] - 8).abs() / 20  # 换手 8% 附近最佳
    df["粗筛得分"] = (
        0.45 * _rank_pct(df["60日涨跌幅"])
        + 0.20 * _rank_pct(df["量比"])
        + 0.20 * turnover_score.clip(0, 1)
        + 0.15 * _rank_pct(df["涨跌幅"])
    )
    df = df.sort_values("粗筛得分", ascending=False).head(pool_size)
    return df[["代码", "名称", "最新价", "涨跌幅", "60日涨跌幅", "总市值", "粗筛得分"]].reset_index(drop=True)


@dataclass
class ScreenResult:
    table: pd.DataFrame          # 最终 Top N 排名表
    n_candidates: int            # 进入精排的候选数
    n_ok: int                    # 成功获取数据并完成评分的数量
    failed: list = field(default_factory=list)  # 数据获取失败的代码
    train_samples: int = 0       # 横截面模型训练样本数


def _make_ranker():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=40,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def fine_rank(
    candidates: pd.DataFrame,
    top_k: int = 100,
    years: float = 1.5,
    sample_step: int = 3,
    horizon: int = 5,
    max_workers: int = 12,
    progress=None,
) -> ScreenResult:
    """阶段二：拉取候选股日线，训练横截面 LightGBM 并输出最终 Top N。

    progress: 可选回调 progress(done, total, stage)，用于 UI 进度条。
    """
    codes = candidates["代码"].tolist()
    names = dict(zip(candidates["代码"], candidates["名称"]))
    total = len(codes)

    # 预热大盘指数缓存，避免线程并发重复拉取
    try:
        model_mod._MARKET_CACHE.setdefault("df", data_mod.fetch_index_history())
    except Exception:
        model_mod._MARKET_CACHE.setdefault("df", None)

    def _fetch(code: str) -> pd.DataFrame:
        try:
            df = data_mod.fetch_history_fast(code, years=years)
        except Exception:
            df = data_mod.fetch_history(code, years=years, prefer_tencent=True)
        if len(df) < 150:
            raise ValueError("历史数据不足")
        return df

    stocks: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, c): c for c in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                stocks[code] = fut.result()
            except Exception:
                failed.append(code)
            done += 1
            if progress:
                progress(done, total, "拉取行情")

    if len(stocks) < 30:
        raise ValueError(f"成功获取数据的股票过少（{len(stocks)}/{total}），请稍后重试")

    # ---- 构造横截面面板 + 各股最新截面 ----
    panel_X, panel_y = [], []
    latest_rows = []
    done = 0
    n_stocks = len(stocks)
    for code, df in stocks.items():
        feats = model_mod.build_features(df)
        feat_cols = list(feats.columns)
        close = df["close"]
        y_fwd = close.shift(-horizon) / close - 1

        valid = feats.notna().sum(axis=1) >= len(feat_cols) - 6  # 容忍少量缺失（LGBM 原生支持 NaN）
        train_mask = valid & y_fwd.notna()
        idx = feats.index[train_mask][::sample_step]
        panel_X.append(feats.loc[idx].values)
        panel_y.append(y_fwd.loc[idx].values)

        try:
            diag_score = signals_mod.diagnose(df).score
        except Exception:
            diag_score = 50
        latest_rows.append(
            {
                "代码": code,
                "名称": names.get(code, code),
                "最新价": float(close.iloc[-1]),
                "20日涨跌": float(close.pct_change(20).iloc[-1]),
                "60日涨跌": float(close.pct_change(60).iloc[-1]) if len(close) > 60 else np.nan,
                "技术诊断分": diag_score,
                "_x": feats.iloc[-1].values,
            }
        )
        done += 1
        if progress:
            progress(done, n_stocks, "构造因子")

    X = np.vstack(panel_X)
    y = np.concatenate(panel_y)
    # 极端值截断（涨停连板等离群样本会扭曲回归目标）
    y = np.clip(y, np.nanquantile(y, 0.005), np.nanquantile(y, 0.995))

    if progress:
        progress(0, 1, "训练横截面模型")
    ranker = _make_ranker()
    ranker.fit(X, y)

    latest = pd.DataFrame(latest_rows)
    latest["AI预测5日收益"] = ranker.predict(np.vstack(latest["_x"].values))
    latest = latest.drop(columns="_x")

    # ---- 综合得分：AI 预测为主，动量与技术面为辅 ----
    mom = latest["20日涨跌"].fillna(0) + latest["60日涨跌"].fillna(0)
    latest["综合得分"] = (
        0.50 * _rank_pct(latest["AI预测5日收益"])
        + 0.25 * _rank_pct(mom)
        + 0.25 * latest["技术诊断分"] / 100
    ) * 100

    latest = latest.sort_values("综合得分", ascending=False).head(top_k).reset_index(drop=True)
    latest.insert(0, "排名", latest.index + 1)
    if progress:
        progress(1, 1, "完成")

    return ScreenResult(
        table=latest,
        n_candidates=total,
        n_ok=n_stocks,
        failed=failed,
        train_samples=len(y),
    )
