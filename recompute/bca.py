"""The finals' statistics, ported from code/finals/stats.py (release branch) so that the recompute scripts need only
numpy: the BCa bootstrap (5,000 resamples, seed 20261004, 95 %, resampling within strata, z0 with ties as half,
jackknife acceleration, linear-interpolated quantiles) and the one-sided non-inferiority bound with the
binomial-floor ceiling guard for bounded per-row differences. Same arithmetic and the same random draws as the
original, so every interval reproduces to the last digit; see that file's docstring for the declarations.
"""
import math
from statistics import NormalDist

import numpy as np

SEED = 20261004
N_RESAMPLES = 5000
LEVEL = 0.95
ALPHA = 0.05
_NORM = NormalDist()


class StatsError(ValueError):
    pass


class IntervalUndefined(StatsError):
    def __init__(self, estimate, n_bad, n_resamples):
        super().__init__(f"the statistic is not finite on {n_bad} of {n_resamples} resamples")
        self.estimate, self.n_bad, self.n_resamples = estimate, n_bad, n_resamples


class Result:
    def __init__(self, estimate, lo, hi, n, degenerate):
        self.estimate, self.lo, self.hi, self.n, self.degenerate = estimate, lo, hi, n, degenerate


def _groups(n, strata):
    if strata is None:
        return [np.arange(n)]
    s = np.asarray([str(x) for x in strata])
    if s.shape != (n,):
        raise StatsError(f"strata has {s.shape[0]} labels for {n} units")
    return [np.flatnonzero(s == lab) for lab in sorted(set(s.tolist()))]


def _estimand_mean(x, groups, weighting):
    if weighting == "unit":
        return x.mean(axis=0)
    return np.mean([x[g].mean(axis=0) for g in groups], axis=0)


def bca(data, *, stat=None, strata=None, weighting="unit", n_resamples=N_RESAMPLES, seed=SEED, level=LEVEL):
    """BCa interval of stat(estimand means) over units; data (n,) or (n, k), columns paired within a unit."""
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, k = x.shape
    if n < 2 or not np.isfinite(x).all():
        raise StatsError(f"BCa needs at least 2 finite units, got {n}")
    stat = stat or (lambda m: m[..., 0])
    groups = _groups(n, strata)
    rng = np.random.default_rng(seed)
    draws = [g[rng.integers(0, len(g), (n_resamples, len(g)))] for g in groups]
    if weighting == "unit":
        idx = np.concatenate(draws, axis=1)
        means_b = np.stack([x[:, j][idx].mean(axis=1) for j in range(k)], axis=-1)
    else:
        means_b = np.mean([np.stack([x[:, j][d].mean(axis=1) for j in range(k)], axis=-1) for d in draws], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_b = np.asarray(stat(means_b), dtype=float)
        theta = float(stat(_estimand_mean(x, groups, weighting)))
    if weighting == "unit":
        loo = (x.sum(axis=0)[None, :] - x) / (n - 1)
    else:
        gm = np.stack([x[g].mean(axis=0) for g in groups])
        loo = np.empty_like(x)
        for gi, g in enumerate(groups):
            rest = gm.sum(axis=0) - gm[gi]
            loo[g] = (rest[None, :] + (x[g].sum(axis=0)[None, :] - x[g]) / (len(g) - 1)) / len(groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_j = np.asarray(stat(loo), dtype=float)
    if not (np.isfinite(theta_b).all() and np.isfinite(theta_j).all() and math.isfinite(theta)):
        raise IntervalUndefined(theta, int(np.count_nonzero(~np.isfinite(theta_b))), n_resamples)
    se = float(theta_b.std(ddof=1))
    d = theta_j.mean() - theta_j
    if se == 0.0 and not d.any():
        return Result(theta, theta, theta, n, True)
    frac = (np.count_nonzero(theta_b < theta) + np.count_nonzero(theta_b <= theta)) / (2 * n_resamples)
    if frac <= 0.0 or frac >= 1.0:
        raise StatsError(f"BCa undefined: the estimate {theta:.6g} lies outside every bootstrap resample")
    z0 = _NORM.inv_cdf(frac)
    ss = float((d ** 2).sum())
    a = float((d ** 3).sum() / (6.0 * ss ** 1.5)) if ss > 0 else 0.0
    z = _NORM.inv_cdf((1.0 - level) / 2.0)
    alphas = [_NORM.cdf(z0 + (z0 + zz) / (1.0 - a * (z0 + zz))) for zz in (z, -z)]
    lo, hi = np.quantile(theta_b, alphas)
    return Result(theta, float(lo), float(hi), n, False)


def binomial_floor(n, alpha=ALPHA):
    return 1.0 - alpha ** (1.0 / n)


def noninferiority_lower(data, *, unit_range=(-100.0, 100.0), alpha=ALPHA, **kw):
    """(estimate, one-sided lower bound, method) of paired differences, for a lower margin: the (1 - 2 alpha)
    BCa interval's lower end, or the exact binomial floor when no unit is worse or the floor is more conservative
    (finals/stats.py noninferiority with unit_range, lower margin only)."""
    r = bca(data, level=1.0 - 2.0 * alpha, **kw)
    d = np.asarray(data, dtype=float)
    floor = binomial_floor(r.n, alpha) * unit_range[0]
    if (d >= 0).all():
        return r.estimate, floor, "exact-binomial-floor"
    if r.degenerate or bool((d == d[:1]).all()):
        raise StatsError("degenerate: zero observed variation is not a finite-sample guarantee")
    if floor < r.lo:
        return r.estimate, floor, "exact-binomial-floor"
    return r.estimate, r.lo, "bca"
