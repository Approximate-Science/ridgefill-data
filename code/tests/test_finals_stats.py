"""finals/stats.py: the finals statistics against independent oracles, with positive and negative controls.

Oracles (BRIEF-COMMON rule 4, simpler than the thing):
  - BCa: scipy.stats.bootstrap(method='BCa') (requirements-dev.txt). (a) Same draws: both sides seeded with the same
    numpy Generator draw the same (n_resamples, n) index matrix, so the endpoints must agree to 1e-9 (implementation
    identity: z0 with ties counted half, the jackknife acceleration, the quantile rule). (b) Independent draws:
    20 seeds per side; the mean endpoint of each side must agree within 4 combined Monte-Carlo standard errors
    (sqrt(var_ours / 20 + var_scipy / 20)); the measured gap and tolerance are printed (-s) and recorded in NOTES.
    Without scipy these tests SKIP by name (never pass).
  - Stratified BCa: a naive per-resample loop written out below, on the same draws (scipy has no strata).
  - MDE: a hand computation on a fixed dataset (the arithmetic is in the test).
  - Holm: a hand-worked family.
Controls: a planted effect is detected and a null is not (and over 200 null datasets the 95% interval misses 0 about
5% of the time); the non-inferiority test passes a clearly equivalent pair and fails a clearly different one, never
passes on a degenerate (zero-variation) bootstrap, and on bounded scores never reports a bound less conservative than
the exact binomial floor (hand-computed: 1 - 0.05^(1/n)); p < 0.05 exactly when the 95% interval excludes 0.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finals"))
import stats as S                                            # noqa: E402

try:
    import scipy.stats as SS
except ImportError:                                          # the repo venv has no scipy: oracle tests skip by name
    SS = None
needs_scipy = pytest.mark.skipif(SS is None, reason="ORACLE SKIPPED: scipy not installed (pip install -r requirements-dev.txt)")

RNG = np.random.default_rng(20261004)
SKEWED19 = RNG.lognormal(0.0, 1.0, 19) - 1.2                # n = 19, skewed: where BCa and percentile differ
A40 = RNG.normal(1.0, 1.0, 40)
B40 = A40 + RNG.normal(0.2, 0.5, 40)
BINARY800 = (RNG.random(800) < 0.55).astype(float) - (RNG.random(800) < 0.45).astype(float)   # RULER-like deltas


def scipy_ci(samples, statistic, seed, paired=False):
    r = SS.bootstrap(samples, statistic, method="BCa", n_resamples=S.N_RESAMPLES, paired=paired, vectorized=True,
                     rng=np.random.default_rng(seed))
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def test_declared_constants():
    assert (S.N_RESAMPLES, S.LEVEL, S.ALPHA, S.POWER, S.SEED) == (5000, 0.95, 0.05, 0.80, 20261004)
    assert S.ESTIMANDS["nll"][0] == "unit" and "document-weighted" in S.ESTIMANDS["nll"][1]
    assert S.ESTIMANDS["accuracy"][0] == "unit" and "sample-weighted" in S.ESTIMANDS["accuracy"][1]
    assert S.ESTIMANDS["longbench"][0] == "stratum"
    with pytest.raises(S.StatsError, match="undeclared estimand"):
        S.weighting_of("token-weighted-nll")


@needs_scipy
@pytest.mark.parametrize("name, samples, ours, statistic, paired", [
    ("mean, skewed n=19", (SKEWED19,), lambda: S.bca(SKEWED19), np.mean, False),
    ("paired delta n=40", (B40, A40), lambda: S.bca(B40 - A40),
     lambda b, a, axis=-1: np.mean(b, axis=axis) - np.mean(a, axis=axis), True),
    ("loss share (ratio) n=40", (A40, B40), lambda: S.bca(np.c_[A40, B40], stat=lambda m: 1 - m[..., 0] / m[..., 1]),
     lambda a, b, axis=-1: 1 - np.mean(a, axis=axis) / np.mean(b, axis=axis), True),
    ("binary RULER-like deltas n=800", (BINARY800,), lambda: S.bca(BINARY800), np.mean, False),
])
def test_bca_equals_scipy_on_the_same_draws(name, samples, ours, statistic, paired):
    r = ours()
    lo, hi = scipy_ci(samples, statistic, S.SEED, paired=paired)
    print(f"\n[same draws] {name}: ours [{r.lo:.10f}, {r.hi:.10f}] scipy [{lo:.10f}, {hi:.10f}]")
    assert abs(r.lo - lo) < 1e-9 and abs(r.hi - hi) < 1e-9


@needs_scipy
@pytest.mark.parametrize("name, data, statistic", [
    ("mean, skewed n=19", SKEWED19, np.mean),
    ("binary RULER-like deltas n=800", BINARY800, np.mean),
])
def test_bca_agrees_with_scipy_on_independent_draws(name, data, statistic):
    seeds = range(20)
    ours = np.array([[S.bca(data, seed=1000 + s).lo, S.bca(data, seed=1000 + s).hi] for s in seeds])
    ref = np.array([scipy_ci((data,), statistic, 2000 + s) for s in seeds])
    se = S.bca(data).se
    for j, end in enumerate(("lo", "hi")):
        gap = abs(ours[:, j].mean() - ref[:, j].mean())
        tol = 4 * math.sqrt(ours[:, j].var(ddof=1) / len(seeds) + ref[:, j].var(ddof=1) / len(seeds))
        print(f"\n[independent draws] {name} {end}: mean ours {ours[:, j].mean():.6f} scipy {ref[:, j].mean():.6f} "
              f"gap {gap:.6f} tolerance {tol:.6f} (MC sd per run ours {ours[:, j].std(ddof=1):.6f}, "
              f"bootstrap SE {se:.6f})")
        assert gap <= tol


def test_stratified_bca_against_a_naive_loop():
    """Oracle: the stratified BCa written out per resample, on the same draws (one rng.integers call per stratum,
    strata in sorted order); also: one stratum equals the unstratified interval."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.3, 1.0, 30)
    strata = ["b"] * 12 + ["a"] * 10 + ["c"] * 8
    B = 2000
    for weighting in ("unit", "stratum"):
        r = S.bca(x, strata=strata, weighting=weighting, n_resamples=B, seed=11)
        # naive oracle
        g = {lab: [i for i, s in enumerate(strata) if s == lab] for lab in sorted(set(strata))}
        gen = np.random.default_rng(11)
        draws = {lab: gen.integers(0, len(ix), (B, len(ix))) for lab, ix in g.items()}

        def est(groups_values):
            if weighting == "unit":
                allv = [v for vs in groups_values.values() for v in vs]
                return sum(allv) / len(allv)
            return sum(sum(vs) / len(vs) for vs in groups_values.values()) / len(groups_values)
        theta = est({lab: [x[i] for i in ix] for lab, ix in g.items()})
        tb = [est({lab: [x[g[lab][j]] for j in draws[lab][b]] for lab in g}) for b in range(B)]
        tj = []
        for i in range(len(x)):
            tj.append(est({lab: [x[k] for k in ix if k != i] for lab, ix in g.items()}))
        frac = (sum(t < theta for t in tb) + sum(t <= theta for t in tb)) / (2 * B)
        z0 = S._NORM.inv_cdf(frac)
        m = sum(tj) / len(tj)
        a = sum((m - t) ** 3 for t in tj) / (6 * sum((m - t) ** 2 for t in tj) ** 1.5)
        al = [S._NORM.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in (-1.959963984540054, 1.959963984540054)]
        lo, hi = np.quantile(tb, al)
        assert abs(r.estimate - theta) < 1e-12 and abs(r.lo - lo) < 1e-9 and abs(r.hi - hi) < 1e-9, weighting
    one = S.bca(x, strata=["s"] * 30, n_resamples=B, seed=11)
    flat = S.bca(x, n_resamples=B, seed=11)
    assert (one.lo, one.hi) == (flat.lo, flat.hi)


def test_mde_hand_oracle():
    # deltas 1..10: mean 5.5, sum of squares about the mean 82.5, sd = sqrt(82.5 / 9) = 3.0276503540974917
    # z(0.975) = 1.959963984540054, z(0.80) = 0.8416212335729143, sum 2.801585218112968
    # MDE = 2.801585218112968 * 3.0276503540974917 / sqrt(10) = 2.682313632510809
    sd, n, m = S.mde_paired(np.arange(1, 11))
    assert n == 10 and abs(sd - 3.0276503540974917) < 1e-12
    assert abs(m - 2.682313632510809) < 1e-12
    # RULER-scale sanity (OUTLINE: per-cell MDE ~25-30 points at 25 samples): sd 50 points, n 25 -> 28.0 points
    assert abs(S.mde(50.0, 25) - 28.01585218112968) < 1e-9
    with pytest.raises(S.StatsError):
        S.mde_paired([1.0])


def test_holm_hand_oracle():
    # sorted: d 0.005 x4 = 0.02; a 0.01 x3 = 0.03; c 0.03 x2 = 0.06; b 0.04 x1 = 0.04 -> monotone max 0.06
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02})
    assert S.holm({"x": 0.6, "y": 0.7}) == pytest.approx({"x": 1.0, "y": 1.0})       # capped at 1
    with pytest.raises(S.StatsError):
        S.holm({"x": 1.5})


def test_planted_effect_is_detected_and_null_is_not():
    rng = np.random.default_rng(3)
    planted = S.bca(rng.normal(0.3, 1.0, 200))
    assert planted.lo > 0 and S.pvalue(planted) < 0.05                       # positive control
    null_data = rng.normal(0.0, 1.0, 200)
    null_data -= null_data.mean()                                            # an exact null sample
    null = S.bca(null_data)
    assert null.lo < 0 < null.hi and S.pvalue(null) > 0.5                    # negative control


def test_null_rejection_rate_is_about_five_percent():
    rng = np.random.default_rng(5)
    rejections = sum(S.bca(rng.normal(0.0, 1.0, 50), seed=s, n_resamples=2000).excludes(0.0) for s in range(200))
    print(f"\n[coverage] 200 null datasets (n=50): 95% BCa interval excluded 0 in {rejections}")
    assert 2 <= rejections <= 20                                             # 5% of 200 = 10; binomial 99.9% band


def test_pvalue_matches_the_interval():
    rng = np.random.default_rng(9)
    for shift in np.linspace(-0.4, 0.4, 17):
        r = S.bca(rng.normal(shift, 1.0, 60))
        p = S.pvalue(r)
        if abs(p - 0.05) > 0.004:                                            # away from the quantile-grid boundary
            assert (p < 0.05) == r.excludes(0.0), (shift, p, r.lo, r.hi)


def test_tost_controls():
    rng = np.random.default_rng(13)
    margin = math.log(1.01)                                                  # ppl +1% as an NLL margin
    same = rng.normal(0.0, 0.002, 40)
    assert S.tost(same, upper=margin).passed                                 # clearly equivalent: passes
    worse = rng.normal(0.03, 0.01, 40)
    assert not S.tost(worse, upper=margin).passed                            # clearly different: fails
    ruler_same = rng.normal(0.0, 3.0, 96)
    assert S.tost(ruler_same, lower=-2.0).passed
    ruler_worse = rng.normal(-6.0, 3.0, 96)
    assert not S.tost(ruler_worse, lower=-2.0).passed
    two_sided = S.tost(same, lower=-margin, upper=margin)
    assert two_sided.passed and two_sided.lo < 0 < two_sided.hi
    t = S.tost(same, upper=margin)
    r90 = S.bca(same, level=0.90)
    assert (t.lo, t.hi) == (r90.lo, r90.hi)                                  # one-sided 95% bounds = the 90% interval
    with pytest.raises(S.StatsError):
        S.tost(same)


def test_noise_floor_degenerate_exact_vs_exact():
    r = S.bca(np.zeros(40))                                                  # deterministic engine: exact == exact
    assert r.degenerate and (r.lo, r.hi, r.estimate) == (0.0, 0.0, 0.0) and S.pvalue(r) == 1.0
    t = S.tost(np.zeros(40), upper=math.log(1.01))                           # ... but never a non-inferiority PASS
    assert not t.passed and t.reason.startswith("degenerate")


def _old_tost(data, lower=None, upper=None, alpha=S.ALPHA, **kw):
    """The pre-fix tost(): the (1 - 2 alpha) BCa interval against the margins (the degenerate path passed)."""
    r = S.bca(data, level=1.0 - 2.0 * alpha, **kw)
    return r.lo, r.hi, (lower is None or r.lo >= lower) and (upper is None or r.hi <= upper)


def test_degenerate_bootstrap_never_certifies_noninferiority():
    t = S.tost(np.zeros(96), lower=-2.0)                                     # the reviewed failure: lo = hi = 0 passed
    assert not t.passed and t.reason == S.DEGENERATE and t.method == "bca"
    r = S.noninferiority(np.full(96, 0.4), lower=-2.0)                       # any identical value, not only 0
    assert not r.passed and r.reason == S.DEGENERATE
    ppl = S.noninferiority(np.zeros(96), upper=math.log1p(0.01))             # perplexity margin: unbounded, all zero
    assert not ppl.passed and ppl.reason.startswith("degenerate")
    worse = S.noninferiority(np.full(500, -1.0), lower=-2.0, unit_range=(-100, 100))   # identical, every unit worse
    assert not worse.passed and worse.reason == S.DEGENERATE


def test_ruler_ceiling_guard_exact_binomial_floor():
    # hand computation: zero regressions in n units bound the share of worse units by 1 - 0.05^(1/n) (Clopper-Pearson,
    # one-sided 95%); a worse unit loses at most 100 points. n = 96: 1 - exp(ln 0.05 / 96) = 0.030724 -> -3.07 points
    for n, want, ok in ((96, -3.07, False), (149, -1.99, True), (150, -1.98, True)):
        r = S.noninferiority(np.zeros(n), lower=-2.0, unit_range=(-100, 100))
        assert round(r.lo, 2) == want and r.passed is ok and r.method == "exact-binomial-floor", (n, r)
        assert r.n == n and r.estimate == 0.0
    assert S.noninferiority(np.zeros(149), lower=-2.0, unit_range=(-100, 100)).lo == pytest.approx(-1.99048, abs=1e-5)
    assert S.noninferiority(np.zeros(150), lower=-2.0, unit_range=(-100, 100)).lo == pytest.approx(-1.97734, abs=1e-5)
    # improvements only (no unit worse): the floor still decides, whatever the improvements
    up = np.zeros(96)
    up[:30] = 100.0
    r = S.noninferiority(up, lower=-2.0, unit_range=(-100, 100))
    assert r.method == "exact-binomial-floor" and round(r.lo, 2) == -3.07 and not r.passed
    # the upper-margin mirror: no unit worse (all <= 0) bounds the mean from above by +floor x b
    r = S.noninferiority(-up, upper=2.0, unit_range=(-100, 100))
    assert r.method == "exact-binomial-floor" and round(r.hi, 2) == 3.07 and not r.passed


def test_ruler_regressions_use_bca_never_less_conservative_than_the_floor():
    floor = lambda n: -100.0 * (1.0 - 0.05 ** (1.0 / n))                     # noqa: E731
    # a few regressions near the margin: 6 of 400 rows lose a needle (mean -1.5); floor(400) = -0.75 points
    d = np.zeros(400)
    d[:6] = -100.0
    r = S.noninferiority(d, lower=-2.0, unit_range=(-100, 100))
    lo, _, _ = _old_tost(d, lower=-2.0)
    assert r.method == "bca" and r.lo == lo and r.lo < floor(400) and r.lo < -2.0 and not r.passed
    # more improvements than regressions: BCa's bound sits above the floor, so the floor is reported
    d = np.zeros(100)
    d[:1], d[1:11] = -100.0, 100.0
    r = S.noninferiority(d, lower=-2.0, unit_range=(-100, 100))
    lo, _, _ = _old_tost(d, lower=-2.0)
    assert lo > floor(100) and r.lo == pytest.approx(floor(100)) and r.method == "exact-binomial-floor"
    assert not r.passed                                                      # floor(100) = -2.95 < -2
    # random RULER-like deltas: the bound is the more conservative of BCa and the floor, every time
    rng = np.random.default_rng(3)
    for _ in range(10):
        n = int(rng.integers(100, 900))
        x = 100.0 * ((rng.random(n) < 0.08).astype(float) - (rng.random(n) < 0.06).astype(float))
        r = S.noninferiority(x, lower=-2.0, unit_range=(-100, 100))
        lo, _, _ = _old_tost(x, lower=-2.0)
        assert r.lo == min(lo, floor(n)) and r.passed == (r.lo >= -2.0)


def test_noninferiority_equals_the_old_tost_on_ordinary_data():
    rng = np.random.default_rng(20261004)
    cases = [(rng.normal(0.0, 0.002, 40), None, math.log(1.01)), (rng.normal(0.0, 3.0, 96), -2.0, None),
             (rng.normal(-6.0, 3.0, 96), -2.0, None), (rng.normal(0.001, 0.003, 60), -0.01, 0.01),
             (BINARY800 * 100, -2.0, None)]
    for x, lower, upper in cases:
        lo, hi, ok = _old_tost(x, lower, upper)
        for r in (S.tost(x, lower=lower, upper=upper), S.noninferiority(x, lower=lower, upper=upper)):
            assert (r.lo, r.hi, r.passed, r.method) == (lo, hi, ok, "bca")
    strata = ["t%d" % (i % 8) for i in range(800)]                           # stratified, as the RULER rows
    near = 100.0 * ((rng.random(800) < 0.45).astype(float) - (rng.random(800) < 0.46).astype(float))
    lo, hi, ok = _old_tost(near, -2.0, strata=strata)
    r = S.noninferiority(near, lower=-2.0, unit_range=(-100, 100), strata=strata)
    assert lo < -100.0 * (1.0 - 0.05 ** (1.0 / 800))
    assert (r.lo, r.hi, r.passed, r.method) == (lo, hi, ok, "bca")          # BCa far below the floor: unchanged


def test_noninferiority_refusals():
    with pytest.raises(S.StatsError, match="margin"):
        S.noninferiority(np.ones(10))
    with pytest.raises(S.StatsError, match="order"):
        S.noninferiority(np.ones(10), lower=1.0, upper=0.0)
    with pytest.raises(S.StatsError, match="outside unit_range"):
        S.noninferiority(np.r_[np.zeros(9), 150.0], lower=-2.0, unit_range=(-100, 100))
    with pytest.raises(S.StatsError, match="per-unit"):
        S.noninferiority(np.zeros((10, 2)), lower=-2.0, unit_range=(-100, 100))
    with pytest.raises(S.StatsError, match="unit-weighted"):
        S.noninferiority(np.zeros(10), lower=-2.0, unit_range=(-100, 100), strata=["a", "b"] * 5, weighting="stratum")
    with pytest.raises(S.StatsError, match="inside"):
        S.noninferiority(np.zeros(10), lower=-2.0, unit_range=(0, 100))


def test_refusals():
    with pytest.raises(S.StatsError, match="unpaired"):
        S.paired({"a": 1.0, "b": 2.0}, {"a": 1.0}, what="x")
    with pytest.raises(S.StatsError, match="unpaired"):
        S.paired({"a": 1.0}, {"a": 1.0, "b": 2.0}, what="x")
    units, xa, xb = S.paired({"a": 1.0}, {"a": 3.0, "b": 2.0}, what="x", subset=True)
    assert units == ["a"] and xb.tolist() == [3.0]
    with pytest.raises(S.StatsError, match="non-finite"):
        S.bca([1.0, float("nan"), 2.0])
    with pytest.raises(S.StatsError, match="at least 2"):
        S.bca([1.0])
    with pytest.raises(S.StatsError, match="one unit"):
        S.bca([1.0, 2.0, 3.0], strata=["a", "a", "b"], weighting="stratum")
    with pytest.raises(S.StatsError, match="not finite"), np.errstate(divide="ignore", invalid="ignore"):
        S.bca(np.c_[np.ones(10), np.zeros(10)], stat=lambda m: m[..., 0] / m[..., 1])


def test_a_ratio_undefined_on_some_resample_has_no_interval():
    """A small gap: the denominator column differs from the reference on 2 of 40 units, so some resamples have a zero
    denominator mean. bca never answers with an interval from the finite resamples: IntervalUndefined (a StatsError)
    carries the finite estimate and the count of non-finite resamples; a zero denominator on the data itself gives a
    non-finite estimate."""
    n = 40
    exact = np.ones(n)
    base = exact.copy()
    base[[3, 17]] = 0.0                                       # the gap to exact: 2 discordant units
    arm = base.copy()
    arm[3] = 1.0                                              # the arm closes one of them
    x = np.column_stack([arm, base, exact])
    gap = lambda m: 100.0 * (m[..., 0] - m[..., 1]) / (m[..., 2] - m[..., 1])          # noqa: E731
    with pytest.raises(S.IntervalUndefined) as e:
        S.bca(x, stat=gap)
    assert isinstance(e.value, S.StatsError) and "not finite on some resample" in str(e.value)
    assert e.value.estimate == pytest.approx(50.0) and 0 < e.value.n_bad < e.value.n_resamples == S.N_RESAMPLES
    assert e.value.n_bad == pytest.approx(S.N_RESAMPLES * (38 / 40) ** 40, rel=0.15)   # P(both gap units missed)
    with pytest.raises(S.IntervalUndefined) as e:
        S.bca(np.column_stack([arm, exact, exact]), stat=gap)
    assert not math.isfinite(e.value.estimate)
    r = S.bca(np.column_stack([arm, np.zeros(n), exact]), stat=gap)                    # a wide gap: an interval
    assert r.lo <= r.estimate <= r.hi
