"""AI 预测模型：基于技术面特征的机器学习多步预测。

思路（直接多步预测法）：
- 从历史行情构造技术面特征（滞后收益率、均线偏离度、RSI、MACD、波动率、量能等）
- 对每个预测步长 h（未来第 1..N 个交易日），单独训练一个模型，
  目标为「未来 h 日累计收益率」，避免递归预测的误差累积
- 用时间序列交叉验证评估方向准确率与误差，并用留出集残差估计置信区间
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import indicators

# lightgbm 在 numpy 输入下偶发的特征名校验提示，无实际影响
warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_CHOICES = {
    "LightGBM (专业·推荐)": "lgbm",
    "Stacking 堆叠融合 (专业)": "stack",
    "模型融合 (Ensemble)": "ensemble",
    "随机森林 (Random Forest)": "rf",
    "梯度提升树 (Gradient Boosting)": "gbdt",
    "神经网络 (MLP)": "mlp",
    "岭回归 (Ridge)": "ridge",
}


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """时序安全的两层 Stacking 融合。

    sklearn 自带的 StackingRegressor 要求 CV 为完整分割（partition），与时序切分不兼容；
    这里手工实现：元特征由 TimeSeriesSplit 的 out-of-fold 预测生成（训练永远在测试之前），
    元学习器用 Ridge 学习各基模型的最优组合权重，全程无未来数据泄漏。
    """

    def __init__(self, kinds: tuple = ("lgbm", "gbdt", "rf"), n_splits: int = 3):
        self.kinds = kinds
        self.n_splits = n_splits

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        oof = np.full((len(y), len(self.kinds)), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            for j, kind in enumerate(self.kinds):
                est = _make_estimator(kind)
                est.fit(X[tr], y[tr])
                oof[te, j] = est.predict(X[te])
        mask = ~np.isnan(oof).any(axis=1)
        self.meta_ = Ridge(alpha=1.0).fit(oof[mask], y[mask])
        self.bases_ = [_make_estimator(k).fit(X, y) for k in self.kinds]
        return self

    def predict(self, X):
        X = np.asarray(X)
        meta_X = np.column_stack([b.predict(X) for b in self.bases_])
        return self.meta_.predict(meta_X)


def _make_estimator(kind: str):
    if kind == "lgbm":
        from lightgbm import LGBMRegressor

        # 针对金融时序低信噪比场景的保守参数：浅树 + 强正则 + 低学习率
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if kind == "stack":
        return TimeSeriesStackingRegressor()
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5, n_jobs=-1, random_state=42
        )
    if kind == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42
        )
    if kind == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=600, early_stopping=True,
            n_iter_no_change=20, random_state=42,
        )
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "ensemble":
        return VotingRegressor(
            [(k, _make_estimator(k)) for k in ("rf", "gbdt", "ridge")], n_jobs=-1
        )
    raise ValueError(f"未知模型类型：{kind}")


def _make_model(kind: str) -> Pipeline:
    return make_pipeline(StandardScaler(), _make_estimator(kind))


# 大盘指数缓存（进程级），保证同一会话内特征列一致
_MARKET_CACHE: dict = {}


def _market_close(dates: pd.Series) -> pd.Series | None:
    """返回与个股交易日对齐的上证指数收盘价序列；获取失败返回 None（特征自动降级）。"""
    if "df" not in _MARKET_CACHE:
        try:
            from . import data as data_mod

            _MARKET_CACHE["df"] = data_mod.fetch_index_history()
        except Exception:
            _MARKET_CACHE["df"] = None
    idx = _MARKET_CACHE["df"]
    if idx is None or idx.empty:
        return None
    s = idx.set_index("date")["close"]
    aligned = s.reindex(pd.to_datetime(dates.values)).ffill()
    aligned.index = dates.index
    return aligned


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 行情构造约 40 维量价/形态/市场因子，返回包含特征列的 DataFrame。"""
    df = indicators.add_all(df)
    out = pd.DataFrame(index=df.index)
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]
    prev_close = close.shift(1)
    ret1 = close.pct_change()

    # ---- 动量 / 均线 ----
    for lag in (1, 2, 3, 5, 10, 20):
        out[f"ret_{lag}d"] = close.pct_change(lag)
    for w in (5, 10, 20, 60):
        out[f"close_ma{w}"] = close / df[f"ma{w}"] - 1

    # ---- 波动率 ----
    out["vol_5d"] = ret1.rolling(5).std()
    out["vol_20d"] = ret1.rolling(20).std()
    out["vol_ratio"] = out["vol_5d"] / out["vol_20d"] - 1  # 短/长期波动比（变盘信号）
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / close

    # ---- 经典振荡指标 ----
    out["rsi14"] = df["rsi14"] / 100
    out["macd_hist"] = df["macd_hist"] / close
    out["kdj_k"] = df["kdj_k"] / 100
    out["boll_pos"] = (close - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"])
    tp = (high + low + close) / 3
    out["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std()) / 100
    hh, ll = high.rolling(14).max(), low.rolling(14).min()
    out["wr_14"] = (hh - close) / (hh - ll)

    # ---- 量能 ----
    out["volume_ratio"] = vol / vol.rolling(5).mean() - 1
    out["vol_trend"] = vol.rolling(5).mean() / vol.rolling(20).mean() - 1
    obv = (np.sign(ret1.fillna(0)) * vol).cumsum()
    out["obv_slope_10"] = (obv - obv.shift(10)) / vol.rolling(10).sum()
    mf = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    out["mfi_14"] = pos_mf / (pos_mf + neg_mf)

    # ---- 收益分布形态 ----
    out["skew_20d"] = ret1.rolling(20).skew()
    out["kurt_20d"] = ret1.rolling(20).kurt()

    # ---- 价格位置（半年高低点） ----
    out["pos_high_120"] = close / close.rolling(120, min_periods=60).max() - 1
    out["pos_low_120"] = close / close.rolling(120, min_periods=60).min() - 1

    # ---- K 线形态 ----
    out["amplitude"] = (high - low) / prev_close
    out["gap"] = open_ / prev_close - 1
    out["body"] = (close - open_) / prev_close
    rng = (high - low).replace(0, np.nan)
    out["upper_shadow"] = (high - np.maximum(close, open_)) / rng
    out["lower_shadow"] = (np.minimum(close, open_) - low) / rng

    # ---- 日历效应 ----
    out["dow"] = (pd.to_datetime(df["date"]).dt.dayofweek - 2) / 2

    # ---- 大盘相对特征（上证指数）----
    mkt = _market_close(df["date"])
    if mkt is not None:
        mret1 = mkt.pct_change()
        out["mkt_ret_1d"] = mret1
        out["mkt_ret_5d"] = mkt.pct_change(5)
        out["mkt_ret_20d"] = mkt.pct_change(20)
        out["mkt_corr_20d"] = ret1.rolling(20).corr(mret1)
        out["beta_60d"] = ret1.rolling(60).cov(mret1) / mret1.rolling(60).var()
        out["rel_strength_20d"] = close.pct_change(20) - mkt.pct_change(20)

    return out.replace([np.inf, -np.inf], np.nan)


@dataclass
class ForecastResult:
    """预测结果与回测评估指标。"""

    forecast: pd.DataFrame  # columns: date, price, lower, upper, cum_return
    direction_accuracy: float  # 次日涨跌方向准确率（时间序列交叉验证）
    baseline_accuracy: float  # 基准：总是预测「涨」的准确率
    mae: float  # 次日收益率平均绝对误差
    feature_importance: pd.Series | None = None
    n_samples: int = 0
    extra: dict = field(default_factory=dict)


def train_and_forecast(
    df: pd.DataFrame,
    horizon: int = 10,
    model_kind: str = "rf",
    n_splits: int = 5,
) -> ForecastResult:
    """训练模型并预测未来 horizon 个交易日的价格走势。"""
    features = build_features(df)
    feat_cols = list(features.columns)
    close = df["close"]

    # ---------- 回测评估（以 h=1 为代表） ----------
    target1 = close.shift(-1) / close - 1
    data1 = pd.concat([features, target1.rename("y")], axis=1).dropna()
    if len(data1) < 120:
        raise ValueError("有效样本不足（少于 120 个交易日），请扩大历史数据范围")

    X1, y1 = data1[feat_cols].values, data1["y"].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dir_hits, base_hits, maes, total = 0, 0, [], 0
    for train_idx, test_idx in tscv.split(X1):
        m = _make_model(model_kind)
        m.fit(X1[train_idx], y1[train_idx])
        pred = m.predict(X1[test_idx])
        actual = y1[test_idx]
        dir_hits += int(np.sum(np.sign(pred) == np.sign(actual)))
        base_hits += int(np.sum(actual > 0))
        maes.append(mean_absolute_error(actual, pred))
        total += len(test_idx)

    # ---------- 逐步长训练 + 预测 ----------
    last_x = features.iloc[[-1]][feat_cols].values
    last_close = float(close.iloc[-1])
    last_date = pd.Timestamp(df["date"].iloc[-1])

    rows = []
    importance_acc = np.zeros(len(feat_cols))
    for h in range(1, horizon + 1):
        target_h = close.shift(-h) / close - 1
        data_h = pd.concat([features, target_h.rename("y")], axis=1).dropna()
        X, y = data_h[feat_cols].values, data_h["y"].values

        # 留出最近 20% 样本估计残差，用于置信区间
        split = int(len(X) * 0.8)
        m_holdout = _make_model(model_kind)
        m_holdout.fit(X[:split], y[:split])
        resid_std = float(np.std(y[split:] - m_holdout.predict(X[split:])))

        m = _make_model(model_kind)
        m.fit(X, y)
        pred_ret = float(m.predict(last_x)[0])

        est = m.steps[-1][1]
        if hasattr(est, "feature_importances_"):
            importance_acc += est.feature_importances_

        rows.append(
            {
                "h": h,
                "cum_return": pred_ret,
                "price": last_close * (1 + pred_ret),
                "lower": last_close * (1 + pred_ret - 1.96 * resid_std),
                "upper": last_close * (1 + pred_ret + 1.96 * resid_std),
            }
        )

    forecast = pd.DataFrame(rows)
    forecast["date"] = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    importance = None
    if importance_acc.sum() > 0:
        importance = (
            pd.Series(importance_acc / horizon, index=feat_cols)
            .sort_values(ascending=False)
        )

    return ForecastResult(
        forecast=forecast,
        direction_accuracy=dir_hits / total,
        baseline_accuracy=base_hits / total,
        mae=float(np.mean(maes)),
        feature_importance=importance,
        n_samples=len(data1),
    )


def walk_forward_replay(
    df: pd.DataFrame,
    model_kind: str = "rf",
    test_size: int = 120,
    retrain_every: int = 10,
) -> pd.DataFrame:
    """滚动回放：模拟「每天只用当天之前的数据训练」，对最近 test_size 个交易日
    逐日预测次日收盘价，与实际值对比，检验模型的真实历史表现。

    返回列：date, actual(实际收盘), predicted(模型当时预测的收盘), pred_ret, actual_ret
    """
    features = build_features(df)
    close = df["close"]
    target = close.shift(-1) / close - 1

    data = pd.concat([features, target.rename("y")], axis=1).dropna()
    if len(data) < test_size + 120:
        test_size = max(30, len(data) - 120)
    if test_size <= 0:
        raise ValueError("样本不足，无法进行滚动回放")

    X, y = data[list(features.columns)].values, data["y"].values
    idx = data.index.to_numpy()
    n = len(data)
    start0 = n - test_size

    preds = np.full(n, np.nan)
    for s in range(start0, n, retrain_every):
        m = _make_model(model_kind)
        m.fit(X[:s], y[:s])
        e = min(s + retrain_every, n)
        preds[s:e] = m.predict(X[s:e])

    rows = idx[start0:]
    base_close = close.loc[rows].to_numpy()  # 预测基准日收盘
    out = pd.DataFrame(
        {
            # 预测目标是「次日」，因此日期取下一交易日
            "date": df["date"].to_numpy()[rows + 1],
            "actual": close.to_numpy()[rows + 1],
            "predicted": base_close * (1 + preds[start0:]),
            "pred_ret": preds[start0:],
            "actual_ret": y[start0:],
        }
    )
    return out
