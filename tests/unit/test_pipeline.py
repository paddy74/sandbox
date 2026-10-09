"""Tests for the pure-Python parts of ``notebooks/pipeline`` (no Spark needed)."""

import datetime as dt
import math

import pandas as pd
from pipeline.config import Config, hav_sql
from pipeline.genie import parse_genie_md
from pipeline.synthetic import designator, generate

NOW = dt.datetime(2026, 1, 1, 12)


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
    ns = {n: getattr(math, n) for n in ["asin", "sqrt", "sin", "cos", "radians"]}
    expr = hav_sql("lat1", "lon1", "lat2", "lon2")

    def dist(lat1, lon1, lat2, lon2):
        return eval(expr, ns | {"lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2})

    arc = 2 * math.pi * 6371000 * 0.01 / 360
    assert math.isclose(dist(39.0, -105.0, 39.01, -105.0), arc)
    assert math.isclose(dist(0.0, -105.0, 0.0, -105.01), arc)


def test_parse_genie_examples() -> None:
    md = Config(catalog="test").repo_root / "demo" / "genie-examples.md"
    instructions, examples = parse_genie_md(md.read_text(), "test.obj_resolution_demo")
    assert "oms_objects" in instructions
    assert [n for n, _, _ in examples] == list(range(1, 16))
    assert all("workspace." not in sql for _, _, sql in examples)
