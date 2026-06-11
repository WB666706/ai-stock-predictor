"""深度学习预测模块：LSTM 与 Transformer 时序模型（PyTorch）。

与 core.model 的逐步长建模不同，深度模型把最近 SEQ_LEN 天的特征序列
作为输入，一次性输出未来 1..horizon 日的累计收益率向量（多输出头），
只需训练一个模型，训练成本可控。

评估方式与经典模型保持一致：
- 按时间顺序留出最近 15% 样本做验证集（无未来数据泄漏）
- 报告 h=1 的方向准确率 / 基准准确率 / MAE
- 用验证集逐步长残差估计 95% 置信区间
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import ForecastResult, build_features

DEEP_MODEL_CHOICES = {
    "LSTM (深度学习)": "lstm",
    "Transformer (深度学习)": "transformer",
}

SEQ_LEN = 30
_SEED = 42


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, horizon)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class _TransformerNet(nn.Module):
    def __init__(self, n_features: int, horizon: int, d_model: int = 64):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, horizon)
        )

    def forward(self, x):
        z = self.embed(x) + self.pos[:, : x.size(1)]
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


def _make_net(kind: str, n_features: int, horizon: int) -> nn.Module:
    if kind == "lstm":
        return _LSTMNet(n_features, horizon)
    if kind == "transformer":
        return _TransformerNet(n_features, horizon)
    raise ValueError(f"未知深度模型类型：{kind}")


def _build_sequences(
    features: pd.DataFrame, close: pd.Series, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 (X 序列, Y 多步收益, 最后一条可用序列 last_x)。"""
    # 多步累计收益目标：第 t 行 -> [t+1..t+h 的累计收益]
    targets = np.stack(
        [(close.shift(-h) / close - 1).to_numpy() for h in range(1, horizon + 1)], axis=1
    )
    feat = features.to_numpy(dtype=np.float64)

    # 仅保留特征完整的行
    valid_feat = ~np.isnan(feat).any(axis=1)
    X_list, y_list = [], []
    n = len(feat)
    for t in range(SEQ_LEN - 1, n):
        window = slice(t - SEQ_LEN + 1, t + 1)
        if not valid_feat[window].all():
            continue
        if np.isnan(targets[t]).any():
            continue
        X_list.append(feat[window])
        y_list.append(targets[t])

    # 最后一条序列（无目标，用于实际预测）
    last_x = None
    t = n - 1
    if valid_feat[t - SEQ_LEN + 1 : t + 1].all():
        last_x = feat[t - SEQ_LEN + 1 : t + 1]

    if not X_list or last_x is None:
        raise ValueError("有效样本不足，无法构造训练序列，请扩大历史数据范围")
    return np.array(X_list), np.array(y_list), last_x


def train_and_forecast_deep(
    df: pd.DataFrame,
    horizon: int = 10,
    model_kind: str = "lstm",
    max_epochs: int = 80,
    patience: int = 10,
) -> ForecastResult:
    """训练深度模型并预测未来 horizon 个交易日，返回与经典模型一致的结果结构。"""
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    features = build_features(df)
    close = df["close"]
    X, y, last_x = _build_sequences(features, close, horizon)
    if len(X) < 150:
        raise ValueError("有效样本不足（少于 150 条序列），请扩大历史数据范围")

    # 按时间留出最近 15% 做验证
    split = int(len(X) * 0.85)
    X_tr, X_va, y_tr, y_va = X[:split], X[split:], y[:split], y[split:]

    # 标准化（仅用训练集统计量）
    mean = X_tr.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8

    def _norm(a: np.ndarray) -> torch.Tensor:
        return torch.tensor((a - mean) / std, dtype=torch.float32)

    Xt_tr, Xt_va = _norm(X_tr), _norm(X_va)
    yt_tr = torch.tensor(y_tr, dtype=torch.float32)
    yt_va = torch.tensor(y_va, dtype=torch.float32)

    net = _make_net(model_kind, X.shape[-1], horizon)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    best_va, best_state, bad = float("inf"), None, 0
    batch = 64
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xt_tr))
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt_tr[idx]), yt_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            va_loss = float(loss_fn(net(Xt_va), yt_va))
        if va_loss < best_va - 1e-6:
            best_va, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    # ---------- 验证集评估（h=1） ----------
    net.eval()
    with torch.no_grad():
        pred_va = net(Xt_va).numpy()
    actual1, pred1 = y_va[:, 0], pred_va[:, 0]
    direction_acc = float(np.mean(np.sign(pred1) == np.sign(actual1)))
    baseline_acc = float(np.mean(actual1 > 0))
    mae = float(np.mean(np.abs(pred1 - actual1)))

    # 逐步长残差 -> 置信区间
    resid_std = np.std(y_va - pred_va, axis=0)

    # ---------- 预测未来 ----------
    with torch.no_grad():
        pred_future = net(_norm(last_x[None, ...])).numpy()[0]

    last_close = float(close.iloc[-1])
    last_date = pd.Timestamp(df["date"].iloc[-1])
    rows = [
        {
            "h": h + 1,
            "cum_return": float(pred_future[h]),
            "price": last_close * (1 + pred_future[h]),
            "lower": last_close * (1 + pred_future[h] - 1.96 * resid_std[h]),
            "upper": last_close * (1 + pred_future[h] + 1.96 * resid_std[h]),
        }
        for h in range(horizon)
    ]
    forecast = pd.DataFrame(rows)
    forecast["date"] = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    return ForecastResult(
        forecast=forecast,
        direction_accuracy=direction_acc,
        baseline_accuracy=baseline_acc,
        mae=mae,
        feature_importance=None,
        n_samples=len(X),
        extra={"val_size": len(X_va), "epochs": epoch + 1},
    )
