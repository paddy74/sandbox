"""Tests for the pure-Python parts of ``notebooks/pipeline`` (no Spark needed)."""

import datetime as dt
import math

import pandas as pd
from pipeline.config import Config, hav_sql
from pipeline.genie import parse_genie_md
from pipeline.icd203 import RANK, likelihood_term
from pipeline.synthetic import (
    GENERIC,
    GENERIC_LABELS,
    N_PARTIAL,
    NOISE_FRAC,
    PARTIAL_SCORE,
    TWIN_M,
    designator,
    generate,
)

NOW = dt.datetime(2026, 1, 1, 12)
_MATH = {n: getattr(math, n) for n in ["asin", "sqrt", "sin", "cos", "radians"]}


def hav_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Evaluate the ``hav_sql`` expression in Python (its functions share math's names)."""
    expr = hav_sql("lat1", "lon1", "lat2", "lon2")
    return eval(expr, _MATH | {"lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2})


def test_designator_wraps_after_zulu() -> None:
    assert [designator(i) for i in (0, 25, 26, 27)] == [
        "Alpha-01",
        "Zulu-01",
        "Alpha-02",
        "Bravo-02",
    ]


def test_generate_is_deterministic() -> None:
    """Cheat-card numbers in docs/plan assume the same seeded data on every run."""
    a, b = generate(NOW), generate(NOW)
    assert len(a.objects) == 2025
    assert a.obs.equals(b.obs) and a.objects.equals(b.objects)
    assert a.reports.equals(b.reports)


def test_reports_entered_after_collection_and_before_now() -> None:
    src = generate(NOW)
    obs_time = pd.to_datetime(src.obs.obs_time)
    report_time = pd.to_datetime(src.reports.report_time)
    assert ((obs_time <= report_time) & (report_time <= NOW)).all()


def test_hav_sql_distances() -> None:
    """0.01 degree along a meridian, or along the equator, is 1,111.95 m on a 6,371 km sphere."""
    arc = 2 * math.pi * 6371000 * 0.01 / 360
    assert math.isclose(hav_m(39.0, -105.0, 39.01, -105.0), arc)
    assert math.isclose(hav_m(0.0, -105.0, 0.0, -105.01), arc)


def test_parse_genie_examples() -> None:
    md = Config(catalog="test").repo_root / "demo" / "genie-examples.md"
    instructions, examples = parse_genie_md(md.read_text(), "test.obj_resolution_demo")
    assert "oms_objects" in instructions
    assert [n for n, _, _ in examples] == list(range(1, 16))
    assert all("workspace." not in sql for _, _, sql in examples)


def test_likelihood_term_band_edges() -> None:
    """Each band includes its lower bound; scores outside 1-99% fall in the outer bands."""
    cases = {0.0: 1, 0.05: 2, 0.5: 4, 0.55: 5, 0.8: 6, 0.95: 7, 1.0: 7}
    assert {p: RANK[likelihood_term(p)] for p in cases} == cases


def test_source_likelihood_follows_producer_rules_and_quality() -> None:
    obs = generate(NOW).obs
    machine = obs[obs.producer == "algorithm"]
    analyst = obs[obs.producer == "analyst"]
    assert (machine.likelihood == machine.confidence.map(likelihood_term)).all()
    assert analyst.confidence.isna().all() and analyst.likelihood.isin(RANK).all()
    cls = (
        pd.Series("exact", index=obs.index)
        .mask(obs.reported_type.isin(GENERIC_LABELS), "generic")
        .mask(obs.reported_type.isna(), "missing")
        .mask(obs.true_object_id.isna(), "noise")
    )
    mean_rank = obs.likelihood.map(RANK).groupby(cls).mean()
    assert list(mean_rank.sort_values(ascending=False).index) == [
        "exact",
        "generic",
        "missing",
        "noise",
    ]


def test_seeded_cases_follow_spec() -> None:
    """Seeded cases as specified in docs/plan/03-data-model.md."""
    src = generate(NOW)
    objects, obs = src.objects.set_index("object_id"), src.obs

    def dist(a, b) -> float:
        return hav_m(a.lat, a.lon, b.lat, b.lon)

    for orig, twin, is_dup in src.twins.itertuples(index=False):
        a, b = objects.loc[orig], objects.loc[twin]
        assert TWIN_M[0] <= dist(a, b) <= TWIN_M[1]
        assert GENERIC[a.object_type] == GENERIC[b.object_type]
        assert a.object_type == b.object_type or not is_dup
    for i in range(N_PARTIAL):
        near_missing, near_generic, low, far = obs[
            obs.true_object_id == f"NEW-{i:04d}"
        ].itertuples()
        assert pd.isna(near_missing.reported_type)
        assert near_generic.reported_type in GENERIC_LABELS
        assert low.producer == "algorithm"
        assert PARTIAL_SCORE[0] <= low.confidence <= PARTIAL_SCORE[1]
        assert dist(far, low) > 400  # beyond the clustering link distance
    noise = obs[obs.true_object_id.isna()]
    assert (noise.producer == "algorithm").all()
    assert math.isclose(len(noise) / len(obs), NOISE_FRAC, abs_tol=0.005)
