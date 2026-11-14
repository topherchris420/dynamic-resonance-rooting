"""Joint-date block inference and self-financing long/cash accounting."""

from __future__ import annotations

import numpy as np


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def block_inference(differences, names, block, replicates, seed):
    """Positive difference is improvement. One common draw preserves model dependence."""
    values = np.asarray(differences, float)
    if values.ndim != 2 or len(values) < block or values.shape[1] != len(names):
        raise ValueError("Invalid inference shape or block length")
    if not np.isfinite(values).all() or replicates < 99:
        raise ValueError("Nonfinite differences or insufficient bootstrap budget")
    rng = np.random.default_rng(seed)
    n = len(values)
    observed = values.mean(axis=0)
    draws = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, 128):
        size = min(128, replicates - start)
        starts = rng.integers(0, n, size=(size, int(np.ceil(n / block))))
        indices = ((starts[..., None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        draws[start : start + size] = values[indices].mean(axis=1)
    centered = draws - observed
    p = (1 + np.sum(centered >= observed, axis=0)) / (replicates + 1)
    # A constant zero difference contains no evidence of improvement.
    p[np.all(values == 0, axis=0)] = 1.0
    adjusted = holm(p)
    bounds = np.quantile(draws, [0.025, 0.975], axis=0)
    return [
        {
            "test": name,
            "block_days": block,
            "mean_improvement": float(observed[i]),
            "ci95_low": float(bounds[0, i]),
            "ci95_high": float(bounds[1, i]),
            "p_one_sided": float(p[i]),
            "p_holm": float(adjusted[i]),
        }
        for i, name in enumerate(names)
    ]


def portfolio(returns, weights, cost_bps):
    """Weights are fractions of post-cost NAV, with residual cash earning zero."""
    r, w = np.asarray(returns, float), np.asarray(weights, float)
    if r.shape != w.shape or r.ndim != 2 or len(r) == 0:
        raise ValueError("Invalid portfolio shape")
    if not np.isfinite(r).all() or not np.isfinite(w).all() or (r <= -1).any():
        raise ValueError("Invalid returns or weights")
    if (w < 0).any() or (w.sum(axis=1) > 1 + 1e-12).any() or not 0 <= cost_bps < 10000:
        raise ValueError("Invalid long/cash weights or costs")
    c = cost_bps / 10000.0
    drifted = np.zeros(r.shape[1])
    net, turnover, costs = [], [], []
    for daily, target in zip(r, w):
        fee = 0.0
        for _ in range(100):
            next_fee = c * float(np.abs((1 - fee) * target - drifted).sum())
            if abs(next_fee - fee) < 1e-14:
                fee = next_fee
                break
            fee = next_fee
        else:
            raise ValueError("Transaction-cost solver did not converge")
        traded = float(np.abs((1 - fee) * target - drifted).sum())
        gross = float(target @ daily)
        net.append((1 - fee) * (1 + gross) - 1)
        turnover.append(traded)
        costs.append(fee)
        drifted = target * (1 + daily) / (1 + gross)
    # Liquidate after the final holding return; fee is in terminal NAV units.
    exit_turnover = float(drifted.sum())
    exit_fee = c * exit_turnover
    terminal_before = 1 + net[-1]
    net[-1] = terminal_before * (1 - exit_fee) - 1
    turnover[-1] += terminal_before * exit_turnover
    costs[-1] += terminal_before * exit_fee
    return {
        "net": np.asarray(net),
        "turnover": np.asarray(turnover),
        "cost": np.asarray(costs),
        "gross": np.sum(r * w, axis=1),
    }


def performance(path, weights):
    net = path["net"]
    equity = np.r_[1.0, np.cumprod(1 + net)]
    sigma = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    return {
        "days": len(net),
        "total_return": float(equity[-1] - 1),
        "cagr": float(equity[-1] ** (365 / len(net)) - 1),
        "mean_daily_net": float(np.mean(net)),
        "annualized_volatility": sigma * np.sqrt(365),
        "sharpe": float(np.mean(net) / sigma * np.sqrt(365)) if sigma > 0 else None,
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "mean_exposure": float(np.asarray(weights).sum(axis=1).mean()),
        "total_turnover": float(path["turnover"].sum()),
        "mean_daily_cost": float(path["cost"].mean()),
    }


def break_even_cost(returns, weights):
    def final(bps):
        return float(np.prod(1 + portfolio(returns, weights, bps)["net"]) - 1)

    if final(0) <= 0:
        return 0.0
    if final(1000) > 0:
        return None  # no root in the declared 0..1000 bps search range
    low, high = 0.0, 1000.0
    for _ in range(35):
        mid = (low + high) / 2
        if final(mid) > 0:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


def prediction_metrics(rows, level=0.9):
    y = rows.target.to_numpy()
    lower, upper = rows.lower.to_numpy(), rows.upper.to_numpy()
    score = (
        upper - lower + (2 / (1 - level)) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    )
    return {
        "rows": len(rows),
        "mse": float(np.mean((rows.prediction.to_numpy() - y) ** 2)),
        "mae": float(np.mean(np.abs(rows.prediction.to_numpy() - y))),
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
        "mean_interval_score": float(np.mean(score)),
    }
