"""Seed-level aggregation, intervals, and power for the token-guess study.

Cycle 1 and the Kelly cycles reported ``mean ± std`` over three seeds. Two
things go wrong with that. The ``±`` reads as an interval but is a population
standard deviation, which for three samples sits about 18% below the sample
standard deviation and roughly 3x below a 95% confidence interval on the mean.
And no comparison was checked for whether three seeds could resolve it, so
orderings were reported for gaps far smaller than the noise around them.

This module supplies the aggregation the study should use instead: a sample
standard deviation, a t-based confidence interval on the mean, a paired
comparison between conditions that share seeds, and the seed count a given
comparison would need before it is worth reporting an ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.95
POWER = 0.80


@dataclass(frozen=True, slots=True)
class Estimate:
    """A condition's score summarised across seeds."""

    mean: float
    sample_sd: float
    n: int
    ci_low: float
    ci_high: float

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


@dataclass(frozen=True, slots=True)
class Comparison:
    """A paired difference between two conditions evaluated on shared seeds."""

    difference: float
    sample_sd: float
    n: int
    ci_low: float
    ci_high: float
    p_value: float
    seeds_for_power: int

    @property
    def resolved(self) -> bool:
        """Whether the interval excludes zero at the configured confidence."""

        return self.ci_low > 0.0 or self.ci_high < 0.0


def summarise(values: Sequence[float], *, confidence: float = CONFIDENCE) -> Estimate:
    """Summarise one condition with a t-based interval on its mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("summarising a condition needs at least two seeds")
    n = len(array)
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sample_sd / np.sqrt(n)
    return Estimate(
        mean=mean,
        sample_sd=sample_sd,
        n=n,
        ci_low=mean - half,
        ci_high=mean + half,
    )


MAX_SEEDS = 10_000


def t_test_power(
    n: int,
    difference: float,
    sample_sd: float,
    *,
    confidence: float = CONFIDENCE,
    paired: bool = True,
) -> float:
    """Power of a two-sided t-test at ``n`` seeds per condition."""

    if n < 2 or sample_sd <= 0.0:
        raise ValueError("power needs at least two seeds and positive spread")
    degrees = n - 1 if paired else 2 * (n - 1)
    scale = np.sqrt(n) if paired else np.sqrt(n / 2.0)
    ncp = abs(difference) / sample_sd * scale
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, degrees))
    return float(
        stats.nct.sf(critical, degrees, ncp) + stats.nct.cdf(-critical, degrees, ncp)
    )


def seeds_for_power(
    difference: float,
    sample_sd: float,
    *,
    power: float = POWER,
    confidence: float = CONFIDENCE,
    paired: bool = True,
) -> int:
    """Seeds per condition needed to resolve a difference of this size.

    Solved against the noncentral t distribution rather than the usual normal
    approximation. At the seed counts this study runs, the two disagree sharply:
    a two-sided test on three seeds has only two degrees of freedom and a
    critical value above four, so the normal approximation can suggest three
    seeds suffice for a comparison that three seeds cannot in fact resolve.
    """

    if sample_sd < 0.0:
        raise ValueError("sample_sd must be non-negative")
    if difference == 0.0:
        return MAX_SEEDS
    if sample_sd == 0.0:
        return 2
    for n in range(2, MAX_SEEDS):
        if t_test_power(
            n,
            difference,
            sample_sd,
            confidence=confidence,
            paired=paired,
        ) >= power:
            return n
    return MAX_SEEDS


def compare(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    confidence: float = CONFIDENCE,
    power: float = POWER,
) -> Comparison:
    """Compare two conditions on the seeds they share.

    Pairing costs nothing when conditions are run on a common seed list and
    removes any variation the seed induces in both conditions at once. It only
    helps to the extent that such shared variation exists; the returned
    ``sample_sd`` is of the per-seed differences, so it reports directly whether
    it did.
    """

    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        raise ValueError("a paired comparison needs at least two shared seeds")
    differences = np.array(
        [left[seed] - right[seed] for seed in shared], dtype=np.float64
    )
    n = len(differences)
    mean = float(differences.mean())
    sample_sd = float(differences.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sample_sd / np.sqrt(n)
    statistic = stats.ttest_rel(
        [left[seed] for seed in shared],
        [right[seed] for seed in shared],
    )
    return Comparison(
        difference=mean,
        sample_sd=sample_sd,
        n=n,
        ci_low=mean - half,
        ci_high=mean + half,
        p_value=float(statistic.pvalue),
        seeds_for_power=seeds_for_power(
            abs(mean),
            sample_sd,
            power=power,
            confidence=confidence,
            paired=True,
        ),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment over a pre-registered comparison family.

    A study that reports every pairwise ordering across eight conditions is
    running 28 tests, and at three seeds several will cross any uncorrected
    threshold by chance. Declare the family up front and adjust within it.
    """

    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * value))
        adjusted[name] = running
    return adjusted
