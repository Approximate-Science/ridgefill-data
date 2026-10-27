#!/usr/bin/env python3
"""Statistics of the RidgeFill finals (paper/PROTOCOL.md section 6, v1.1). Only finals/tables.py calls this; no
number reaches the paper without passing through here.

Declared before any final is read (the constants below are the declaration; tests/test_finals_stats.py pins them):
  BCa bootstrap   bias-corrected and accelerated (Efron 1987), N_RESAMPLES = 5,000 resamples of units (a unit =
                  one prompt / document / RULER row with all its arms: the pairing unit), seed SEED, LEVEL 95%.
                  Locked sets resample rows within their stratum (task; documents within domain) when the caller
                  passes strata. z0 counts ties as half (scipy's rule); the acceleration comes from the
                  leave-one-unit-out jackknife; the interval endpoints are linear-interpolated quantiles of the
                  bootstrap distribution. Oracle: scipy.stats.bootstrap(method='BCa') (tests).
  estimands       ESTIMANDS: NLL is document-weighted (every document counts once, whatever its token count),
                  accuracy is sample-weighted (every RULER row counts once, pooled over tasks and lengths),
                  LongBench is task-weighted (the mean of the task means, LongBench's own average), decode
                  agreement and KL are prompt-weighted. A contrast names its estimand; nothing else is computed.
  p-values        two-sided, by inverting the BCa interval (the level at which the interval's endpoint reaches the
                  null), so p < 0.05 exactly when the 95% BCa interval excludes the null (up to quantile
                  interpolation). Used only as Holm's multiplicity guard, never as evidence on its own.
  Holm            step-down over one declared family of secondary contrasts.
  MDE             (z(1 - ALPHA/2) + z(POWER)) x SD(paired deltas) / sqrt(n): 80% power, two-sided 5%, normal
                  approximation, unstratified (stated as such in the paper).
  non-inferiority every declared criterion is a one-sided margin (perplexity upper +1%, RULER lower -2 points), so it
                  is a NON-INFERIORITY test, not two-sided equivalence: the one-sided (1 - ALPHA) BCa bound (one end
                  of the (1 - 2 ALPHA) interval) must sit on the right side of its margin (a bound equal to its
                  margin does not exceed it). Two declared margins make it two one-sided tests (TOST, equivalence).
                  A degenerate bootstrap (every paired difference identical) is not evidence: unbounded data then
                  never pass. Bounded paired scores (unit_range, e.g. RULER per-row accuracy differences in
                  [-100, 100] points) get a ceiling guard: when no unit is worse than the reference in the margin's
                  direction, the bound is the exact one-sided binomial floor (1 - ALPHA^(1/n)) x (the most a unit can
                  lose), the Clopper-Pearson bound on the share of worse units times the worst loss (96 units with
                  no regression: -3.07 points); otherwise the BCa bound, never less conservative than that floor.
                  Margins come from the FINAL- record (finals/tables.py), never from here.

Refusals are StatsError with the reason (unpaired units, non-finite values, too few units, a BCa that does not
exist); a statistic that is not finite on some resample is the subclass IntervalUndefined, which carries the estimate. Nothing returns an approximate answer silently. Pure numpy + the standard library (statistics.NormalDist);
scipy is the oracle in the tests only (requirements-dev.txt).
"""
import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

SEED = 20261004                      # the finals' declared bootstrap seed (OUTLINE T1.7 uses the same date seed)
N_RESAMPLES = 5000                   # OUTLINE revision 2: "BCa bootstrap, 5,000 resamples"
LEVEL = 0.95                         # two-sided interval level
ALPHA = 0.05                         # test level (two-sided for MDE and p-values; one-sided for non-inferiority)
POWER = 0.80                         # MDE power
_NORM = NormalDist()

# estimand -> (weighting, definition). "unit": every unit counts once; "stratum": the mean of the stratum means.
ESTIMANDS = {
    "nll": ("unit", "document-weighted: mean over documents of each document's arm-minus-exact mean NLL"),
    "accuracy": ("unit", "sample-weighted: mean over RULER rows (each row once, pooled over tasks and lengths)"),
    "longbench": ("stratum", "task-weighted: mean over the tasks of each task's mean row score (LongBench's average)"),
    "agreement": ("unit", "prompt-weighted: mean over prompts of each prompt's share of agreeing decode steps"),
    "kl": ("unit", "prompt-weighted: mean over prompts of each prompt's mean per-step KL(exact || arm)"),
}


class StatsError(ValueError):
    """A named refusal: the statistic cannot be computed from these inputs as declared."""


class IntervalUndefined(StatsError):
    """The statistic is not finite on some bootstrap resample or leave-one-out set (a ratio whose denominator mean is
    zero there), so no BCa interval exists. Never answered with an interval from the finite resamples (that would be
    another, undeclared estimator). estimate: the statistic on the data (it may itself be non-finite); n_bad: the
    resamples where it is not finite. A caller that reports the estimate alone prints no interval and says why."""

    def __init__(self, estimate, n_bad, n_resamples):
        super().__init__("the statistic is not finite on some resample (e.g. a ratio over a zero mean): "
                         f"{n_bad} of {n_resamples} resamples")
        self.estimate, self.n_bad, self.n_resamples = estimate, n_bad, n_resamples


def weighting_of(estimand):
    if estimand not in ESTIMANDS:
        raise StatsError(f"undeclared estimand {estimand!r}: declared are {sorted(ESTIMANDS)}")
    return ESTIMANDS[estimand][0]


def paired(a, b, *, what, subset=False):
    """Pair two {unit: value} maps. Returns (units, xa, xb), units sorted.

    The unit sets must be identical (the pairing unit holds every arm); subset=True allows b to hold more units than
    a (a population restricted on a's side, e.g. question-first prompts against the question-last arm) but never
    fewer. Non-finite values are refused."""
    ka, kb = set(a), set(b)
    if not ka:
        raise StatsError(f"{what}: no units on the first side")
    missing_b = sorted(ka - kb)
    extra_b = sorted(kb - ka)
    if missing_b or (extra_b and not subset):
        raise StatsError(f"{what}: unpaired units (first side only: {missing_b[:5]}{'...' if len(missing_b) > 5 else ''}; "
                         f"second side only: {extra_b[:5]}{'...' if len(extra_b) > 5 else ''}): the pairing unit "
                         "must hold every arm of the contrast")
    units = sorted(ka)
    xa = np.array([a[u] for u in units], dtype=float)
    xb = np.array([b[u] for u in units], dtype=float)
    if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
        raise StatsError(f"{what}: non-finite value in a paired unit")
    return units, xa, xb


def _groups(n, strata):
    if strata is None:
        return [np.arange(n)]
    s = np.asarray([str(x) for x in strata])
    if s.shape != (n,):
        raise StatsError(f"strata has {s.shape[0]} labels for {n} units")
    return [np.flatnonzero(s == lab) for lab in sorted(set(s.tolist()))]


def _estimand_mean(x, groups, weighting):
    """(k,) estimand means of the (n, k) data."""
    if weighting == "unit":
        return x.mean(axis=0)
    return np.mean([x[g].mean(axis=0) for g in groups], axis=0)


@dataclass
class BCa:
    estimate: float
    lo: float
    hi: float
    n: int
    se: float
    z0: float
    a: float
    level: float
    n_resamples: int
    seed: int
    degenerate: bool
    theta_b: np.ndarray = field(repr=False)

    def excludes(self, null=0.0):
        return self.lo > null or self.hi < null


def bca(data, *, stat=None, strata=None, weighting="unit", n_resamples=N_RESAMPLES, seed=SEED, level=LEVEL):
    """BCa interval of stat(estimand means) over units.

    data: (n,) or (n, k) per-unit values (paired columns of one unit stay together in every resample).
    stat: function of the (..., k) estimand means -> (...); default the first column (a paired delta).
    strata: per-unit labels; resampling is within each stratum (the locked sets' rows within task).
    weighting: "unit" (every unit once) or "stratum" (mean of the stratum means); see ESTIMANDS.

    Degenerate data (every resample and every leave-one-out value equal, e.g. identical paired differences) return
    the zero-width interval [theta, theta] with degenerate=True. That is exact for deterministic diagnostics (an
    exact-vs-exact rerun on a deterministic engine), never a finite-sample guarantee: noninferiority() refuses to
    pass on it."""
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise StatsError(f"data must be (n,) or (n, k), got shape {x.shape}")
    n, k = x.shape
    if n < 2:
        raise StatsError(f"BCa needs at least 2 units, got {n}")
    if not np.isfinite(x).all():
        raise StatsError("non-finite value in the data")
    if weighting not in ("unit", "stratum"):
        raise StatsError(f"unknown weighting {weighting!r}")
    if not (0.0 < level < 1.0) or n_resamples < 100:
        raise StatsError(f"level {level} / n_resamples {n_resamples} out of range")
    stat = stat or (lambda m: m[..., 0])
    groups = _groups(n, strata)
    if weighting == "stratum" and min(len(g) for g in groups) < 2:
        raise StatsError("a stratum with one unit has no leave-one-out mean (stratum weighting)")

    rng = np.random.default_rng(seed)
    draws = [g[rng.integers(0, len(g), (n_resamples, len(g)))] for g in groups]   # one call per stratum, sorted
    if weighting == "unit":
        idx = np.concatenate(draws, axis=1)
        means_b = np.stack([x[:, j][idx].mean(axis=1) for j in range(k)], axis=-1)
    else:
        means_b = np.mean([np.stack([x[:, j][d].mean(axis=1) for j in range(k)], axis=-1) for d in draws], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):         # a non-finite statistic is refused below, by name
        theta_b = np.asarray(stat(means_b), dtype=float)
        theta = float(stat(_estimand_mean(x, groups, weighting)))

    # leave-one-unit-out estimand means (n, k), closed form
    if weighting == "unit":
        loo = (x.sum(axis=0)[None, :] - x) / (n - 1)
    else:
        gm = np.stack([x[g].mean(axis=0) for g in groups])            # (G, k)
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
        return BCa(theta, theta, theta, n, 0.0, 0.0, 0.0, level, n_resamples, seed, True, theta_b)
    frac = (np.count_nonzero(theta_b < theta) + np.count_nonzero(theta_b <= theta)) / (2 * n_resamples)
    if frac <= 0.0 or frac >= 1.0:
        raise StatsError(f"BCa undefined: the estimate {theta:.6g} lies outside every bootstrap resample")
    z0 = _NORM.inv_cdf(frac)
    ss = float((d ** 2).sum())
    a = float((d ** 3).sum() / (6.0 * ss ** 1.5)) if ss > 0 else 0.0
    z = _NORM.inv_cdf((1.0 - level) / 2.0)
    alphas = [_NORM.cdf(z0 + (z0 + zz) / (1.0 - a * (z0 + zz))) for zz in (z, -z)]
    lo, hi = np.quantile(theta_b, alphas)                          # linear interpolation (scipy's default)
    return BCa(theta, float(lo), float(hi), n, se, z0, a, level, n_resamples, seed, False, theta_b)


def pvalue(res, null=0.0):
    """Two-sided p-value of H0: theta = null from the BCa interval inverted at null (floor 1 / n_resamples)."""
    floor = 1.0 / res.n_resamples
    if res.degenerate:
        return 1.0 if res.estimate == null else floor
    B = res.n_resamples
    g = (np.count_nonzero(res.theta_b < null) + np.count_nonzero(res.theta_b <= null)) / (2 * B)
    g = min(max(g, 1.0 / (2 * B)), 1.0 - 1.0 / (2 * B))
    v = _NORM.inv_cdf(g) - res.z0
    if 1.0 + res.a * v <= 0.0:                                     # beyond every reachable BCa endpoint
        return floor
    tail = _NORM.cdf(v / (1.0 + res.a * v) - res.z0)               # the endpoint level that reaches null
    return float(min(1.0, max(floor, 2.0 * min(tail, 1.0 - tail))))


def holm(pvals):
    """Holm-adjusted p-values of one family {name: p}: step-down, monotone, capped at 1."""
    names = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(names)
    out, run = {}, 0.0
    for i, name in enumerate(names):
        p = pvals[name]
        if not (0.0 <= p <= 1.0):
            raise StatsError(f"Holm: p-value of {name!r} is {p}")
        run = max(run, min(1.0, (m - i) * p))
        out[name] = run
    return out


def mde(sd, n, alpha=ALPHA, power=POWER):
    """Minimum detectable effect of a paired mean: (z(1 - alpha/2) + z(power)) * sd / sqrt(n)."""
    if n < 2 or not math.isfinite(sd) or sd < 0:
        raise StatsError(f"MDE needs n >= 2 and a finite sd >= 0 (n {n}, sd {sd})")
    return (_NORM.inv_cdf(1.0 - alpha / 2.0) + _NORM.inv_cdf(power)) * sd / math.sqrt(n)


def mde_paired(deltas, alpha=ALPHA, power=POWER):
    """(sd, n, mde) of per-unit paired deltas (sd with ddof 1)."""
    d = np.asarray(deltas, dtype=float)
    if d.ndim != 1 or d.size < 2 or not np.isfinite(d).all():
        raise StatsError("MDE needs at least 2 finite paired deltas")
    sd = float(d.std(ddof=1))
    return sd, int(d.size), mde(sd, d.size, alpha, power)


@dataclass
class NonInferiority:
    estimate: float
    lo: float                        # one-sided (1 - alpha) lower bound (the reported bound of a lower margin)
    hi: float                        # one-sided (1 - alpha) upper bound (the reported bound of an upper margin)
    lower: object                    # declared lower margin or None
    upper: object                    # declared upper margin or None
    passed: bool
    n: int
    alpha: float
    method: str                      # "bca" or "exact-binomial-floor" (the method of the bound(s) tested)
    reason: str


TOST = NonInferiority                # the earlier name of the result


DEGENERATE = "degenerate: zero observed variation is not a finite-sample guarantee"


def binomial_floor(n, alpha=ALPHA):
    """Exact one-sided (1 - alpha) upper bound on the share of units worse than the reference when none of n is:
    1 - alpha^(1/n) (Clopper-Pearson with zero events)."""
    return 1.0 - alpha ** (1.0 / n)


def noninferiority(data, *, lower=None, upper=None, alpha=ALPHA, unit_range=None, **kw):
    """Non-inferiority of paired differences against declared one-sided margins (both margins: TOST equivalence).

    Passes when (lower is None or lo >= lower) and (upper is None or hi <= upper), lo / hi the one-sided (1 - alpha)
    bounds: the (1 - 2 alpha) BCa interval's ends (kw go to bca(): strata, weighting, ...).
    unit_range=(a, b), a < 0 < b: the range of one unit's paired difference (RULER per-row accuracy difference in
    points: (-100, 100)). When no unit is worse in a margin's direction (all differences >= 0 for a lower margin,
    <= 0 for an upper one) that bound is the exact binomial floor (1 - alpha^(1/n)) * a (resp. * b), valid for any
    distribution of the improvements; otherwise the BCa bound, or the floor when the floor is more conservative.
    Degenerate data (all differences identical) never pass unless the floor decides (no unit worse, bounded data)."""
    if lower is None and upper is None:
        raise StatsError("non-inferiority needs at least one declared margin")
    if lower is not None and upper is not None and not lower < upper:
        raise StatsError(f"non-inferiority margins out of order: {lower} >= {upper}")
    if not (0.0 < alpha < 0.5):
        raise StatsError(f"alpha {alpha} out of range")
    r = bca(data, level=1.0 - 2.0 * alpha, **kw)
    x = np.asarray(data, dtype=float)
    # identical units: bca() flags exact zero spread only, and float rounding of a constant's mean can hide it
    degenerate = r.degenerate or bool((x == x[:1]).all())
    lo, hi, methods, reasons = r.lo, r.hi, set(), []
    if unit_range is None:
        if degenerate:
            return NonInferiority(r.estimate, lo, hi, lower, upper, False, r.n, alpha, "bca", DEGENERATE)
        methods.add("bca")
        reasons.append(f"one-sided {100 * (1 - alpha):g}% BCa bound")
    else:
        d = x
        a, b = (float(v) for v in unit_range)
        if d.ndim != 1:
            raise StatsError("unit_range applies to (n,) per-unit paired differences only")
        if kw.get("weighting", "unit") != "unit":
            raise StatsError("unit_range's binomial floor holds for unit-weighted means only")
        if not a < 0.0 < b:
            raise StatsError(f"unit_range {unit_range} must hold 0 strictly inside")
        if d.min() < a or d.max() > b:
            raise StatsError(f"a paired difference lies outside unit_range {unit_range}")
        share = binomial_floor(r.n, alpha)
        for side, margin in (("lower", lower), ("upper", upper)):
            if margin is None:
                continue
            floor = share * (a if side == "lower" else b)
            none_worse = bool((d >= 0).all() if side == "lower" else (d <= 0).all())
            if none_worse:
                bound, m = floor, "exact-binomial-floor"
                reasons.append(f"no unit worse than the reference: exact binomial floor, "
                               f"{100 * share:.2f}% worse units at most")
            elif degenerate:
                return NonInferiority(r.estimate, lo, hi, lower, upper, False, r.n, alpha, "bca", DEGENERATE)
            else:
                bca_bound = r.lo if side == "lower" else r.hi
                more = (floor < bca_bound) if side == "lower" else (floor > bca_bound)
                bound, m = (floor, "exact-binomial-floor") if more else (bca_bound, "bca")
                reasons.append(f"one-sided {100 * (1 - alpha):g}% BCa bound" + (", replaced by the more conservative "
                               "exact binomial floor" if more else ""))
            if side == "lower":
                lo = bound
            else:
                hi = bound
            methods.add(m)
    ok = (lower is None or lo >= lower) and (upper is None or hi <= upper)
    method = methods.pop() if len(methods) == 1 else "+".join(sorted(methods))
    return NonInferiority(r.estimate, float(lo), float(hi), lower, upper, bool(ok), r.n, alpha, method,
                          "; ".join(reasons))


def tost(data, *, lower=None, upper=None, alpha=ALPHA, **kw):
    """Non-inferiority test (equivalence when both margins are declared): noninferiority() under its earlier name.

    The declared criteria are one-sided margins (C5's ppl upper margin; RULER's lower margin), so these are
    non-inferiority tests; a degenerate interval never passes (noninferiority())."""
    return noninferiority(data, lower=lower, upper=upper, alpha=alpha, **kw)
