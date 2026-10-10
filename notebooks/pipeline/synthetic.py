"""Synthetic sources (Layer 1): object system records, observations, reports and tasking."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .icd203 import TERMS, likelihood_term

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from .config import Config

# Object types are CCO-native labels; each GENERIC value is a CCO ancestor of its key
# (checked against the ontology in ontology.load_cco).
LEAVES = [
    "Airport",
    "Military Facility",
    "Truck",
    "Armored Fighting Vehicle",
    "Aircraft",
]
GENERIC = {
    "Airport": "Facility",
    "Military Facility": "Facility",
    "Truck": "Ground Vehicle",
    "Armored Fighting Vehicle": "Ground Vehicle",
    "Aircraft": "Vehicle",
}
GENERIC_LABELS = sorted(set(GENERIC.values()))

# Human-readable designator: CCO Arbitrary Identifier, linked by "designates" and stored as its text value.
DESIGNATOR_IRI = "https://www.commoncoreontologies.org/ont00000923"
PHONETIC = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India",
            "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
            "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray", "Yankee", "Zulu"]  # fmt: skip

N_OBJ, N_OBSERVED, N_NEW = 2000, 1500, 150
# Twin pairs 150-400 m apart; the first N_DUP are duplicate records of one real object.
N_TWIN, N_DUP = 25, 5
TWIN_M = (150, 400)
# N_PARTIAL of the N_NEW missing objects are seen only through individually unusable observations.
N_PARTIAL, N_CROSS = 30, 100
PARTIAL_SCORE = (0.40, 0.55)  # machine score of the partial, specific-type observation
FAR_M = (1000, 3000)  # position error of the partial object's km-off observation
CROSS_M = 100  # cross-producer observations fall within this distance of the object
NOISE_FRAC = 0.05  # share of all observations with no real object behind them
N_TASKS = 40
# min_lat, max_lat, min_lon, max_lon of an arbitrary area.
BBOX = (39.0, 39.5, -105.5, -105.0)
SENSORS = ["EO", "SAR", "FMV"]
# Minutes from collection to the record being entered: (low, high, weight) ranges per producer.
REPORT_DELAY_MIN = {
    "algorithm": [(1, 5, 0.6), (5, 30, 0.3), (30, 120, 0.1)],  # machine: mostly minutes
    "analyst": [
        (30, 120, 0.3),
        (120, 480, 0.5),
        (480, 1440, 0.2),
    ],  # hours, up to a day
}
assert all(math.isclose(sum(w for *_, w in r), 1) for r in REPORT_DELAY_MIN.values()), (
    "REPORT_DELAY_MIN weights must sum to 1 per producer"
)

# The source's own likelihood that its report is right, lower for noise and for missing or
# generic types. Machines give a score drawn uniformly from (low, high); analysts give an
# ICD 203 term drawn by weight.
MACHINE_SCORE = {
    "exact": (0.70, 0.99),
    "generic": (0.45, 0.85),
    "missing": (0.25, 0.70),
    "noise": (0.05, 0.60),
}
ANALYST_TERM = {
    "exact": {"Likely": 0.3, "Very likely": 0.5, "Almost certain": 0.2},
    "generic": {"Roughly even chance": 0.3, "Likely": 0.5, "Very likely": 0.2},
    "missing": {"Unlikely": 0.3, "Roughly even chance": 0.4, "Likely": 0.3},
    "noise": {
        "Very unlikely": 0.2,
        "Unlikely": 0.4,
        "Roughly even chance": 0.3,
        "Likely": 0.1,
    },
}
assert all(
    set(w) <= set(TERMS) and math.isclose(sum(w.values()), 1)
    for w in ANALYST_TERM.values()
), "ANALYST_TERM must use ICD 203 terms with weights summing to 1"


@dataclass
class Sources:
    """Everything the generator produces, before it is written anywhere.

    ``obs.true_object_id`` and ``twins`` are ground truth for scoring the demo, never features.
    """

    objects: pd.DataFrame
    twins: pd.DataFrame
    obs: pd.DataFrame
    reports: pd.DataFrame
    tasking: pd.DataFrame


def designator(i: int) -> str:
    """Unique call-sign style designator for the i-th object, e.g. ``Bravo-12``."""
    return f"{PHONETIC[i % len(PHONETIC)]}-{i // len(PHONETIC) + 1:02d}"


def jitter(
    rng: np.random.Generator, lat: float, lon: float, sigma_m: float
) -> tuple[float, float]:
    """Point moved by Gaussian noise with standard deviation ``sigma_m`` metres."""
    dlat = rng.normal(0, sigma_m) / 111_320
    dlon = rng.normal(0, sigma_m) / (111_320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def offset(
    lat: float, lon: float, dist_m: float, bearing_deg: float
) -> tuple[float, float]:
    """Point ``dist_m`` metres from (lat, lon) on a bearing."""
    b = math.radians(bearing_deg)
    return (
        lat + dist_m * math.cos(b) / 111_320,
        lon + dist_m * math.sin(b) / (111_320 * math.cos(math.radians(lat))),
    )


def report_delays(rng: np.random.Generator, producers: pd.Series) -> np.ndarray:
    """Reporting delay per observation, from its producer's ``REPORT_DELAY_MIN`` ranges.

    Each draw picks a range by weight, then a uniform value within it.

    :return: delays in minutes, aligned with ``producers``.
    :raises ValueError: for a producer with no delay ranges.
    """
    unknown = set(producers) - set(REPORT_DELAY_MIN)
    if unknown:
        raise ValueError(f"no REPORT_DELAY_MIN ranges for producers {sorted(unknown)}")
    delays = np.empty(len(producers))
    for producer, ranges in REPORT_DELAY_MIN.items():
        mask = (producers == producer).to_numpy()
        low, high, weight = (np.array(x) for x in zip(*ranges, strict=True))
        idx = rng.choice(len(ranges), mask.sum(), p=weight)
        delays[mask] = rng.uniform(low[idx], high[idx])
    return delays


def assign_likelihood(
    rng: np.random.Generator, obs: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """The source's own assessment of each observation, lower when it got less right.

    A machine (``algorithm``) produces a numeric score, mapped to its ICD 203 term; an analyst
    produces only a term, so their score is ``None``. Classes: ``noise`` (no real object),
    ``missing`` (no type), ``generic`` (a CCO ancestor type) or ``exact``. Observations whose
    ``case`` is ``partial_low`` get a machine score in ``PARTIAL_SCORE`` instead.

    :return: (confidence score or ``None``, likelihood term), aligned with ``obs``.
    """
    cls = np.select(
        [
            obs.true_object_id.isna().to_numpy(),
            obs.reported_type.isna().to_numpy(),
            obs.reported_type.isin(GENERIC_LABELS).to_numpy(),
        ],
        ["noise", "missing", "generic"],
        "exact",
    )
    machine = (obs.producer == "algorithm").to_numpy()
    score = pd.Series(None, index=obs.index, dtype=object)
    term = pd.Series(None, index=obs.index, dtype=object)
    for c in MACHINE_SCORE:  # fixed order, so draws are reproducible
        m = machine & (cls == c)
        low, high = MACHINE_SCORE[c]
        drawn = np.round(rng.uniform(low, high, m.sum()), 2)
        score[m] = drawn.tolist()
        term[m] = [likelihood_term(x) for x in drawn]
        a = ~machine & (cls == c)
        weights = ANALYST_TERM[c]
        term[a] = rng.choice(list(weights), a.sum(), p=list(weights.values())).tolist()
    low_spec = (obs.case == "partial_low").to_numpy()
    drawn = np.round(rng.uniform(*PARTIAL_SCORE, low_spec.sum()), 2)
    score[low_spec] = drawn.tolist()
    term[low_spec] = [likelihood_term(x) for x in drawn]
    return score, term


def make_obs(
    rng: np.random.Generator,
    now: dt.datetime,
    obj_id: str | None,
    otype: str | None,
    lat: float,
    lon: float,
    k: int,
    producers: list[str] | None = None,
    exact: bool = False,
    weak: bool = False,
    within_m: float | None = None,
    case: str | None = None,
) -> list[dict]:
    """``k`` observations of one real object (or of nothing, if ``obj_id`` is ``None``).

    Times fall in the 72 hours before ``now``. Positions scatter around (lat, lon) by
    producer (sigma 80 m for algorithms, 150 m for analysts) unless ``within_m`` is set.

    :param producers: one producer per observation; default is random, 70% algorithm.
    :param exact: always report ``otype`` as given (``None`` reports no type).
    :param weak: never report the specific type (generic or missing, 50/50).
    :param within_m: place each observation uniformly within this many metres instead.
    :param case: seeded-case tag read by ``assign_likelihood``; dropped before landing.
    :return: observation rows, with ground truth in ``true_object_id``.
    """
    rows = []
    for j in range(k):
        producer = (
            producers[j]
            if producers
            else rng.choice(["algorithm", "analyst"], p=[0.7, 0.3])
        )
        if within_m is None:
            olat, olon = jitter(rng, lat, lon, 80 if producer == "algorithm" else 150)
        else:
            olat, olon = offset(lat, lon, rng.uniform(0, within_m), rng.uniform(0, 360))
        r = rng.random()
        if exact:
            rtype = otype
        elif weak:
            rtype = None if r < 0.5 else GENERIC[otype]
        else:
            rtype = otype if r < 0.7 else (GENERIC[otype] if r < 0.9 else None)
        rows.append(
            {
                "obs_id": f"OBS-{rng.integers(1e12):012d}",
                "obs_time": (
                    now - dt.timedelta(hours=float(rng.uniform(0, 72)))
                ).isoformat(),
                "lat": olat,
                "lon": olon,
                "reported_type": rtype,
                "producer": producer,
                "sensor": str(rng.choice(SENSORS)),
                "true_object_id": obj_id,
                "case": case,
            }
        )
    return rows


def generate(now: dt.datetime, seed: int = 42) -> Sources:
    """Build the seeded synthetic sources.

    Seeded cases, as specified in ``docs/plan/03-data-model.md``, make the review queue real:
    **twin** object records 150-400 m apart, **duplicates** of one real object, **partial**
    observations of objects missing from the object system, **noise** with no object behind
    it and **cross-producer** corroboration. Changing the order of random draws changes every
    downstream number.

    :param now: reference time; observations fall in the 72 hours before it.
    :param seed: seed for the main random stream.
    """
    rng = np.random.default_rng(seed)
    objects = pd.DataFrame(
        {
            "object_id": [f"OBJ-{i:05d}" for i in range(N_OBJ)],
            "object_type": rng.choice(LEAVES, N_OBJ),
            "lat": rng.uniform(BBOX[0], BBOX[1], N_OBJ),
            "lon": rng.uniform(BBOX[2], BBOX[3], N_OBJ),
            "marking": rng.choice(["OPEN", "RESTRICTED"], N_OBJ, p=[0.8, 0.2]),
            "dup_of": None,
        }
    )

    # Twins: a second record 150-400 m from an existing one. A duplicate has the same type;
    # a look-alike has a compatible type (same generic CCO ancestor, e.g. Truck and AFV).
    twin_src = objects.sample(N_TWIN, random_state=3).reset_index(drop=True)
    twin_pairs, twin_rows = [], []
    for i, o in twin_src.iterrows():
        is_dup = i < N_DUP
        tlat, tlon = offset(o.lat, o.lon, rng.uniform(*TWIN_M), rng.uniform(0, 360))
        compatible = [t for t in LEAVES if GENERIC[t] == GENERIC[o.object_type]]
        tw = {
            "object_id": f"OBJ-T{i:03d}",
            "object_type": o.object_type if is_dup else str(rng.choice(compatible)),
            "lat": tlat,
            "lon": tlon,
            "marking": o.marking,
            "dup_of": o.object_id if is_dup else None,
        }
        twin_rows.append(tw)
        twin_pairs.append((o.to_dict(), tw, is_dup))
    objects = pd.concat([objects, pd.DataFrame(twin_rows)], ignore_index=True)
    objects["first_seen"] = now - dt.timedelta(days=30)
    objects["obs_count"] = 0
    objects["source"] = "OMS"
    objects["status"] = "ACTIVE"
    objects["designator"] = [designator(i) for i in range(len(objects))]
    objects["designator_type"] = DESIGNATOR_IRI
    assert objects.designator.is_unique, "designators must be unique"

    def obs_of(obj_id, otype, lat, lon, k, **kw):
        return make_obs(rng, now, obj_id, otype, lat, lon, k, **kw)

    obs_rows = []
    for _, o in objects.sample(N_OBSERVED, random_state=1).iterrows():
        obs_rows += obs_of(
            o.object_id, o.object_type, o.lat, o.lon, int(rng.integers(1, 5))
        )
    # Objects missing from the object system: these should become nominations.
    for i in range(N_NEW):
        oid, otype = f"NEW-{i:04d}", str(rng.choice(LEAVES))
        lat, lon = rng.uniform(BBOX[0], BBOX[1]), rng.uniform(BBOX[2], BBOX[3])
        if i >= N_PARTIAL:
            obs_rows += obs_of(oid, otype, lat, lon, int(rng.integers(2, 4)))
            continue
        # Partial: none is usable alone. A missing and a generic type plus a specific type
        # with a low machine score cluster into one nomination; a fourth, about a kilometre
        # off, stays unclustered to show the pipeline does not over-merge.
        # With exact=True the given type is the one reported: none, then the generic one.
        obs_rows += obs_of(oid, None, lat, lon, 1, exact=True)
        obs_rows += obs_of(oid, GENERIC[otype], lat, lon, 1, exact=True)
        obs_rows += obs_of(oid, otype, lat, lon, 1, producers=["algorithm"], exact=True,
                           case="partial_low")  # fmt: skip
        far_lat, far_lon = offset(lat, lon, rng.uniform(*FAR_M), rng.uniform(0, 360))
        obs_rows += obs_of(oid, otype, far_lat, far_lon, 1, exact=True)
    for i, (o, tw, is_dup) in enumerate(twin_pairs):
        # Between the two records, generic or missing type. A duplicate's observations all
        # belong to the original; a look-alike pair's alternate between the two.
        t = o if (is_dup or i % 2 == 0) else tw
        f = rng.uniform(0.2, 0.8, 3)
        for x in f:
            lat = o["lat"] + x * (tw["lat"] - o["lat"])
            lon = o["lon"] + x * (tw["lon"] - o["lon"])
            obs_rows += obs_of(t["object_id"], t["object_type"], lat, lon, 1, weak=True)
    for _, o in objects.sample(N_CROSS, random_state=5).iterrows():
        obs_rows += obs_of(o.object_id, o.object_type, o.lat, o.lon, 2,
                           producers=["algorithm", "analyst"], exact=True,
                           within_m=CROSS_M)  # fmt: skip
    # Noise: single algorithm detections with no object behind them, placed at random.
    n_noise = round(NOISE_FRAC * len(obs_rows) / (1 - NOISE_FRAC))
    for _ in range(n_noise):
        lat, lon = rng.uniform(BBOX[0], BBOX[1]), rng.uniform(BBOX[2], BBOX[3])
        obs_rows += obs_of(None, str(rng.choice(LEAVES)), lat, lon, 1,
                           producers=["algorithm"], exact=True)  # fmt: skip

    obs = pd.DataFrame(obs_rows)
    obs["batch"] = rng.integers(1, 3, len(obs))  # delivered in two files

    # Activity history: objects that are still being observed were seen more recently.
    # Separate generators here and below, so tuning one does not change the observations.
    rng_seen = np.random.default_rng(seed + 1)
    active = objects.object_id.isin(obs.true_object_id)
    days_ago = np.where(
        active,
        rng_seen.uniform(1, 14, len(objects)),
        rng_seen.uniform(1, 30, len(objects)),
    )
    objects.insert(
        objects.columns.get_loc("first_seen") + 1,
        "last_seen",
        [now - dt.timedelta(days=float(d)) for d in days_ago],
    )
    print(
        f"objects={len(objects)} (twins={N_TWIN}, duplicates={N_DUP}) observations={len(obs)} "
        f"(generic type={obs.reported_type.isin(GENERIC_LABELS).sum()}, "
        f"missing type={obs.reported_type.isna().sum()}, noise={n_noise})"
    )

    score, term = assign_likelihood(np.random.default_rng(seed + 3), obs)
    obs.insert(obs.columns.get_loc("reported_type") + 1, "confidence", score)
    obs.insert(obs.columns.get_loc("confidence") + 1, "likelihood", term)
    obs = obs.drop(columns="case")

    # Reports are entered after collection, never after now: a delay that would overshoot is
    # redrawn uniformly within the time left.
    rng_report = np.random.default_rng(seed + 2)
    obs_ts = pd.to_datetime(obs.obs_time)
    delay = pd.to_timedelta(report_delays(rng_report, obs.producer), unit="min")
    room = now - obs_ts
    delay = delay.where(delay <= room, room * rng_report.uniform(0, 1, len(obs)))
    report_time = (obs_ts + delay).dt.ceil("s").map(lambda t: t.isoformat())

    # Templated narrative standing in for finished reports, one per observation.
    obs["report_id"] = "RPT-" + obs.obs_id.str[4:]
    reports = pd.DataFrame(
        {
            "report_id": obs.report_id,
            "obs_id": obs.obs_id,
            "report_time": report_time,
            "report_text": [
                f"{r.producer.title()} report ({r.sensor}): {r.reported_type or 'unidentified object'} "
                f"observed at {r.lat:.4f}, {r.lon:.4f} on {r.obs_time[:16].replace('T', ' ')}Z. "
                f"Source assessment: {r.likelihood.lower()}."
                for r in obs.itertuples()
            ],
        }
    )

    # Collection tasking: sensor coverage boxes and windows (mock MySQL export).
    tasking = pd.DataFrame(
        {
            "task_id": [f"TASK-{i:03d}" for i in range(N_TASKS)],
            "sensor": rng.choice(SENSORS, N_TASKS),
            "min_lat": (la := rng.uniform(BBOX[0], BBOX[1] - 0.05, N_TASKS)),
            "max_lat": la + 0.05,
            "min_lon": (lo := rng.uniform(BBOX[2], BBOX[3] - 0.05, N_TASKS)),
            "max_lon": lo + 0.05,
            "start_time": [
                (now - dt.timedelta(hours=float(h))).isoformat()
                for h in rng.uniform(24, 72, N_TASKS)
            ],
            "end_time": [
                (now - dt.timedelta(hours=float(h))).isoformat()
                for h in rng.uniform(0, 24, N_TASKS)
            ],
            "requested_by": rng.choice(
                ["ANALYSIS-TEAM", "COMMAND-A", "COMMAND-B"], N_TASKS
            ),
            "priority": rng.integers(1, 4, N_TASKS),
        }
    )
    twins = pd.DataFrame(
        [(o["object_id"], tw["object_id"], bool(d)) for o, tw, d in twin_pairs],
        columns=["original_id", "twin_id", "is_duplicate"],
    )
    return Sources(objects, twins, obs, reports, tasking)


def write_jsonl(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame as JSON lines to a (volume) path; missing values become ``null``."""
    with open(path, "w") as f:
        for r in df.astype(object).where(df.notna(), None).to_dict("records"):
            f.write(json.dumps(r) + "\n")


def write_sources(spark: SparkSession, cfg: Config, src: Sources) -> dict:
    """Load the object system into ``oms_objects`` and drop the other sources as landing files.

    Observation batch 2 goes to the ``staged`` folder; ``ingest.ingest`` delivers it after
    loading batch 1 to show incremental loading. Run ``ingest.reset`` first.

    :return: cheat-card facts.
    """
    spark.createDataFrame(src.objects).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{cfg.s}.oms_objects")
    spark.createDataFrame(src.twins).write.mode("overwrite").saveAsTable(
        f"{cfg.s}.seed_twins"
    )
    obs = src.obs.drop(columns=["report_id"])
    write_jsonl(
        obs[obs.batch == 1].drop(columns="batch"),
        f"{cfg.landing['observations']}/batch_1.json",
    )
    write_jsonl(
        obs[obs.batch == 2].drop(columns="batch"),
        f"{cfg.landing['staged']}/batch_2.json",
    )
    write_jsonl(src.reports, f"{cfg.landing['reports']}/reports_1.json")
    src.tasking.to_csv(f"{cfg.landing['tasking']}/tasking_1.csv", index=False)
    return {"observations": len(src.obs), "oms_objects_seeded": len(src.objects)}
