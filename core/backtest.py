"""策略回测引擎：把模型的次日收益预测变成模拟交易策略，并与买入持有对比。

策略规则（T+1 简化模拟）：
- 滚动训练：每隔 retrain_every 个交易日，用「截至当日」的全部历史数据重新训练模型
- 当模型预测次日收益 > 阈值时持有（满仓），否则空仓
- 换仓时扣除交易成本（按单边费率）

注意：回测全程不使用未来数据（每次预测仅用过去样本训练）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import _make_model, build_features


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # date / strategy / buy_hold / position
    metrics: dict  # 各项绩效指标
    n_trades: int


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1))


def run_backtest(
    df: pd.DataFrame,
    model_kind: str = "rf",
    test_size: int = 250,
    retrain_every: int = 20,
    threshold: float = 0.0,
    cost: float = 0.001,
    sizing: str = "binary",
) -> BacktestResult:
    """对最近 test_size 个交易日做模型信号策略回测。

    sizing:
    - "binary": 预测为正则满仓、否则空仓
    - "confidence": 置信度加权——仓位 = 预测收益 / 历史波动，预测越强仓位越高（0~100%）
    """
    features = build_features(df)
    close = df["close"]
    target = close.shift(-1) / close - 1

    data = pd.concat([features, target.rename("y")], axis=1).dropna()
    if len(data) < 180:
        raise ValueError("有效样本不足（少于 180 个交易日），请扩大历史数据范围")
    test_size = min(test_size, len(data) - 120)

    X, y = data[list(features.columns)].values, data["y"].values
    n = len(data)
    start0 = n - test_size

    preds = np.full(n, np.nan)
    sigmas = np.full(n, np.nan)  # 截至训练时点的历史日收益波动（用于置信度归一化）
    for s in range(start0, n, retrain_every):
        m = _make_model(model_kind)
        m.fit(X[:s], y[:s])
        e = min(s + retrain_every, n)
        preds[s:e] = m.predict(X[s:e])
        sigmas[s:e] = np.std(y[:s])

    pred_test = preds[start0:]
    ret_test = y[start0:]  # 持仓当日实际获得的次日收益

    if sizing == "confidence":
        position = np.clip(pred_test / (1.5 * sigmas[start0:]), 0.0, 1.0)
        position[pred_test <= threshold] = 0.0
    else:
        position = (pred_test > threshold).astype(float)

    # 换仓成本：position 变化的当日按单边费率扣除
    switches = np.abs(np.diff(position, prepend=0.0))
    strat_ret = position * ret_test - switches * cost

    equity_strat = np.cumprod(1 + strat_ret)
    equity_hold = np.cumprod(1 + ret_test)

    idx = data.index.to_numpy()[start0:]
    dates = df["date"].to_numpy()[idx + 1]  # 收益在次日实现

    holding_days = int(np.sum(position > 0))
    win_days = int(np.sum((ret_test > 0) & (position > 0)))
    years_frac = test_size / 252

    def _annual(eq: np.ndarray) -> float:
        return float(eq[-1] ** (1 / years_frac) - 1)

    std = float(np.std(strat_ret))
    sharpe = float(np.mean(strat_ret) / std * np.sqrt(252)) if std > 0 else 0.0

    metrics = {
        "策略总收益": float(equity_strat[-1] - 1),
        "买入持有总收益": float(equity_hold[-1] - 1),
        "策略年化收益": _annual(equity_strat),
        "买入持有年化": _annual(equity_hold),
        "策略最大回撤": _max_drawdown(equity_strat),
        "买入持有最大回撤": _max_drawdown(equity_hold),
        "夏普比率": sharpe,
        "持仓天数占比": holding_days / test_size,
        "持仓日胜率": win_days / holding_days if holding_days else 0.0,
    }

    equity = pd.DataFrame(
        {
            "date": dates,
            "strategy": equity_strat,
            "buy_hold": equity_hold,
            "position": position,
        }
    )
    return BacktestResult(
        equity=equity,
        metrics=metrics,
        n_trades=int(np.sum(switches > 0)),
    )
