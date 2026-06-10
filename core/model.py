"""AI 预测模型：基于技术面特征的机器学习多步预测。

思路（直接多步预测法）：
- 从历史行情构造技术面特征（滞后收益率、均线偏离度、RSI、MACD、波动率、量能等）
- 对每个预测步长 h（未来第 1..N 个交易日），单独训练一个模型，
  目标为「未来 h 日累计收益率」，避免递归预测的误差累积
- 用时间序列交叉验证评估方向准确率与误差，并用留出集残差估计置信区间
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
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

MODEL_CHOICES = {
    "模型融合 (Ensemble)": "ensemble",
    "随机森林 (Random Forest)": "rf",
    "梯度提升树 (Gradient Boosting)": "gbdt",
    "神经网络 (MLP)": "mlp",
    "岭回归 (Ridge)": "ridge",
}


def _make_estimator(kind: str):
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 行情构造模型特征，返回包含特征列的 DataFrame。"""
    df = indicators.add_all(df)
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    for lag in (1, 2, 3, 5, 10, 20):
        out[f"ret_{lag}d"] = close.pct_change(lag)

    for w in (5, 10, 20, 60):
        out[f"close_ma{w}"] = close / df[f"ma{w}"] - 1

    ret1 = close.pct_change()
    out["vol_5d"] = ret1.rolling(5).std()
    out["vol_20d"] = ret1.rolling(20).std()

    out["rsi14"] = df["rsi14"] / 100
    out["macd_hist"] = df["macd_hist"] / close
    out["kdj_k"] = df["kdj_k"] / 100
    out["boll_pos"] = (close - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"])

    vol_ma5 = df["volume"].rolling(5).mean()
    out["volume_ratio"] = df["volume"] / vol_ma5 - 1
    out["amplitude"] = (df["high"] - df["low"]) / close.shift(1)
    return out


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
