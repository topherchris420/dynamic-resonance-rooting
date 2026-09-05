"""
Quantitative ML Prediction & Signal Evaluation Metrics (IC, Rank IC, ICIR, Ablation).
"""

from typing import Dict, Any, Sequence, Optional
import numpy as np
import pandas as pd
from scipy import stats


def calculate_ic_metrics(
    predictions: pd.Series,
    labels: pd.Series,
) -> Dict[str, float]:
    """
    Calculate Information Coefficient (IC), Rank IC, ICIR, Rank ICIR, and RMSE.
    """
    valid_mask = predictions.notna() & labels.notna()
    pred_clean = predictions[valid_mask]
    label_clean = labels[valid_mask]

    if len(pred_clean) < 10:
        return {
            "ic": 0.0,
            "ic_pvalue": 1.0,
            "rank_ic": 0.0,
            "rank_ic_pvalue": 1.0,
            "icir": 0.0,
            "rank_icir": 0.0,
            "rmse": 0.0,
        }

    # Pearson Information Coefficient
    ic, ic_pval = stats.pearsonr(pred_clean, label_clean)

    # Spearman Rank Information Coefficient
    rank_ic, rank_ic_pval = stats.spearmanr(pred_clean, label_clean)

    # Calculate rolling or grouped ICs for ICIR if time series indexed
    if isinstance(pred_clean.index, pd.DatetimeIndex):
        df_eval = pd.DataFrame({"pred": pred_clean, "label": label_clean})
        grouped_ic = (
            df_eval.groupby(df_eval.index.to_period("M"))
            .apply(lambda g: stats.pearsonr(g["pred"], g["label"])[0] if len(g) > 3 else np.nan)
            .dropna()
        )
        ic_mean = float(grouped_ic.mean()) if len(grouped_ic) > 0 else ic
        ic_std = float(grouped_ic.std()) if len(grouped_ic) > 1 and grouped_ic.std() > 1e-8 else 1.0
        icir = float(ic_mean / ic_std) if ic_std > 1e-8 else 0.0

        grouped_rank_ic = (
            df_eval.groupby(df_eval.index.to_period("M"))
            .apply(lambda g: stats.spearmanr(g["pred"], g["label"])[0] if len(g) > 3 else np.nan)
            .dropna()
        )
        rank_ic_mean = float(grouped_rank_ic.mean()) if len(grouped_rank_ic) > 0 else rank_ic
        rank_ic_std = (
            float(grouped_rank_ic.std())
            if len(grouped_rank_ic) > 1 and grouped_rank_ic.std() > 1e-8
            else 1.0
        )
        rank_icir = float(rank_ic_mean / rank_ic_std) if rank_ic_std > 1e-8 else 0.0
    else:
        icir = float(ic)
        rank_icir = float(rank_ic)

    rmse = float(np.sqrt(np.mean((pred_clean - label_clean) ** 2)))

    return {
        "ic": float(ic),
        "ic_pvalue": float(ic_pval),
        "rank_ic": float(rank_ic),
        "rank_ic_pvalue": float(rank_ic_pval),
        "icir": icir,
        "rank_icir": rank_icir,
        "rmse": rmse,
    }
