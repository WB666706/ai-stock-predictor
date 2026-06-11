"""专业量化评估体系：Purged 时序交叉验证、IC/RankIC/ICIR、分位数收益分析、方向概率。

这些是专业量化机构评估预测信号质量的标准方法：
- Purged CV：训练集与测试集之间留出 embargo 空窗，消除标签重叠造成的隐性泄漏
  （参考 Marcos López de Prado《Advances in Financial Machine Learning》）
- IC (Information Coefficient)：预测值与实际收益的截面相关性，> 0.03 即有商业价值
- RankIC：Spearman 秩相关，对异常值更稳健
- ICIR：IC 均值 / IC 标准差，衡量信号的稳定性
- 分位数分析：按预测值分组看实际收益是否单调递增——好信号应该「预测越高、实际越涨」
- 方向概率：独立的分类模型输出「次日上涨概率」，比点估计更可操作
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .model import _make_model, build_features


def purged_splits(n: int, n_splits: int = 5, embargo: int = 5, min_train: int = 120):
    """带 embargo 空窗的时序切分：训练集末尾与测试集开头隔开 embargo 个样本。"""
    fold = (n - min_train) // n_splits
    if fold <= 0:
        raise ValueError("样本不足，无法进行 Purged 交叉验证")
    for i in range(n_splits):
        start = min_train + i * fold
        end = start + fold if i < n_splits - 1 else n
        yield np.arange(0, max(1, start - embargo)), np.arange(start, end)


def _make_classifier():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            random_state=42, n_jobs=-1, verbosity=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        )


@dataclass
class ProEvalResult:
    ic: float                    # Pearson IC（OOF 预测 vs 实际次日收益）
    rank_ic: float               # Spearman RankIC
    icir: float                  # 按折 IC 均值 / 标准差
    direction_auc: float         # 方向分类 AUC（0.5 为随机水平）
    prob_up: float               # 最新交易日的「次日上涨概率」
    quantile_table: pd.DataFrame # 预测分位 vs 实际平均收益 / 胜率
    oof: pd.DataFrame            # date / pred / actual（全部 OOF 样本）
    n_samples: int


def professional_eval(
    df: pd.DataFrame,
    model_kind: str = "lgbm",
    n_splits: int = 5,
    embargo: int = 5,
) -> ProEvalResult:
    """对次日收益预测信号做专业级评估（全程 out-of-fold，无未来数据）。"""
    features = build_features(df)
    feat_cols = list(features.columns)
    close = df["close"]
    target = close.shift(-1) / close - 1

    data = pd.concat([features, target.rename("y")], axis=1).dropna()
    if len(data) < 180:
        raise ValueError("有效样本不足（少于 180 个交易日），请扩大历史数据范围")

    X, y = data[feat_cols].values, data["y"].values
    y_cls = (y > 0).astype(int)
    n = len(data)

    oof_pred = np.full(n, np.nan)
    oof_prob = np.full(n, np.nan)
    fold_ics = []
    for train_idx, test_idx in purged_splits(n, n_splits=n_splits, embargo=embargo):
        m = _make_model(model_kind)
        m.fit(X[train_idx], y[train_idx])
        p = m.predict(X[test_idx])
        oof_pred[test_idx] = p
        if np.std(p) > 0 and np.std(y[test_idx]) > 0:
            fold_ics.append(float(stats.pearsonr(p, y[test_idx])[0]))

        clf = _make_classifier()
        clf.fit(X[train_idx], y_cls[train_idx])
        oof_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    mask = ~np.isnan(oof_pred)
    pred_v, act_v, prob_v = oof_pred[mask], y[mask], oof_prob[mask]

    ic = float(stats.pearsonr(pred_v, act_v)[0])
    rank_ic = float(stats.spearmanr(pred_v, act_v)[0])
    icir = float(np.mean(fold_ics) / np.std(fold_ics)) if len(fold_ics) > 1 and np.std(fold_ics) > 0 else 0.0
    cls_v = (act_v > 0).astype(int)
    auc = float(roc_auc_score(cls_v, prob_v)) if 0 < cls_v.sum() < len(cls_v) else 0.5

    # ---- 分位数分析：预测值分 5 组，看实际收益是否单调 ----
    q = pd.qcut(pred_v, 5, labels=False, duplicates="drop")
    qt = (
        pd.DataFrame({"q": q, "actual": act_v})
        .groupby("q")["actual"]
        .agg(平均实际收益="mean", 胜率=lambda s: float((s > 0).mean()), 样本数="count")
        .reset_index()
    )
    qt["分位"] = qt["q"].map(lambda i: f"Q{int(i) + 1}{'（最看空）' if i == 0 else ('（最看多）' if i == qt['q'].max() else '')}")
    quantile_table = qt[["分位", "平均实际收益", "胜率", "样本数"]]

    # ---- 最新一日的次日上涨概率（用全部样本训练）----
    clf_full = _make_classifier()
    clf_full.fit(X, y_cls)
    last_x = features.iloc[[-1]][feat_cols].values
    prob_up = float(clf_full.predict_proba(last_x)[0, 1])

    dates = df.loc[data.index, "date"].to_numpy()[mask]
    oof = pd.DataFrame({"date": dates, "pred": pred_v, "actual": act_v, "prob_up": prob_v})

    return ProEvalResult(
        ic=ic,
        rank_ic=rank_ic,
        icir=icir,
        direction_auc=auc,
        prob_up=prob_up,
        quantile_table=quantile_table,
        oof=oof,
        n_samples=int(mask.sum()),
    )
