"""Weighted least squares with several sets of fixed effects, and the standard errors it needs.

Fixed effects are swept out by alternating projections, so a panel with a million rows and
hundreds of thousands of effects fits in memory. Two covariance estimators are provided because
the errors here are dependent in two ways: cells that share a zone move together, and every cell
in a week shares that week's traffic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def codes(*columns: pd.Series) -> np.ndarray:
    """Integer codes for the interaction of one or more columns."""
    if len(columns) == 1:
        return pd.factorize(columns[0])[0]
    return (
        pd.DataFrame(dict(enumerate(columns)))
        .groupby(list(range(len(columns))), sort=False, observed=True)
        .ngroup()
        .to_numpy()
    )


def drop_singletons(groups: list[np.ndarray]) -> np.ndarray:
    """Rows that survive after repeatedly removing groups with a single row (they carry no
    information once their own effect is fitted, and understate standard errors if kept)."""
    keep = np.ones(len(groups[0]), bool)
    while True:
        before = keep.sum()
        for g in groups:
            counts = np.bincount(g[keep], minlength=g.max() + 1)
            keep &= counts[g] > 1
        if keep.sum() == before:
            return keep


def demean(
    x: np.ndarray, groups: list[np.ndarray], w: np.ndarray, tol: float = 1e-9, max_iter: int = 2000
) -> np.ndarray:
    x = np.array(x, dtype=float, copy=True)
    if x.ndim == 1:
        x = x[:, None]
    totals = [np.bincount(g, w) for g in groups]
    for col in range(x.shape[1]):
        v = x[:, col]
        for _ in range(max_iter):
            change = 0.0
            for g, tot in zip(groups, totals, strict=True):
                means = np.bincount(g, w * v) / tot
                change = max(change, np.abs(means).max())
                v -= means[g]
            if change < tol:
                break
        else:
            raise RuntimeError(f"fixed effects did not converge (last change {change:.2e})")
    return x


@dataclass
class Fit:
    names: list[str]
    coef: np.ndarray
    cov_cluster: np.ndarray
    cov_time: np.ndarray
    n: int

    @property
    def se_cluster(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov_cluster), 0))

    @property
    def se_time(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov_time), 0))

    @property
    def se(self) -> np.ndarray:
        """The larger of the two: neither source of dependence is ignored."""
        return np.maximum(self.se_cluster, self.se_time)

    def contrast(self, weights: dict[str, float]) -> tuple[float, float]:
        """A linear combination of coefficients and its (conservative) standard error."""
        c = np.array([weights.get(n, 0.0) for n in self.names])
        var = max(c @ self.cov_cluster @ c, c @ self.cov_time @ c)
        return float(c @ self.coef), float(np.sqrt(var))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.coef,
                "se": self.se,
                "se_cluster": self.se_cluster,
                "se_time": self.se_time,
            },
            index=self.names,
        )


def _meat(score: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, int]:
    sums = np.stack([np.bincount(group, score[:, j]) for j in range(score.shape[1])], axis=1)
    return sums.T @ sums, len(np.unique(group))


def _driscoll_kraay(score: np.ndarray, time: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West on the time series of cross-sectional score sums."""
    h = np.stack([np.bincount(time, score[:, j]) for j in range(score.shape[1])], axis=1)
    s = h.T @ h
    for lag in range(1, lags + 1):
        gamma = h[lag:].T @ h[:-lag]
        s += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    return s


def fit(
    y: np.ndarray,
    x: pd.DataFrame,
    groups: list[np.ndarray],
    w: np.ndarray,
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
    time: np.ndarray,
    lags: int = 8,
) -> Fit:
    """`time` must be ordered integer codes. Clusters are the two zones of each trip."""
    keep = drop_singletons(groups)
    y, w = np.asarray(y, float)[keep], np.asarray(w, float)[keep]
    xm = x.to_numpy(float)[keep]
    groups = [pd.factorize(g[keep])[0] for g in groups]
    # work on a unit-mean weight scale so the tolerance means the same thing in every model
    w = w / w.mean()
    yd = demean(y, groups, w)[:, 0]
    xd = demean(xm, groups, w)
    bread = np.linalg.inv(xd.T @ (w[:, None] * xd))
    coef = bread @ (xd.T @ (w * yd))
    score = xd * (w * (yd - xd @ coef))[:, None]
    a, b = pd.factorize(cluster_a[keep])[0], pd.factorize(cluster_b[keep])[0]
    ab = codes(pd.Series(a), pd.Series(b))
    ma, na = _meat(score, a)
    mb, nb = _meat(score, b)
    mab, _ = _meat(score, ab)
    g = min(na, nb)
    cov_cluster = bread @ (ma + mb - mab) @ bread * g / (g - 1)
    t = pd.factorize(time[keep], sort=True)[0]
    nt = t.max() + 1
    cov_time = bread @ _driscoll_kraay(score, t, lags) @ bread * nt / (nt - 1)
    return Fit(list(x.columns), coef, cov_cluster, cov_time, len(y))
