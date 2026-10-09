"""Synthetic sources (Layer 1): object system records, observations, reports and tasking."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
# Twin pairs; the first N_DUP are true duplicates of one real object, the rest look-alikes.
N_TWIN, N_DUP = 25, 5
N_PARTIAL, N_CROSS = 30, 100
NOISE_FRAC = 0.05
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


def make_obs(
    rng: np.random.Generator,
    now: dt.datetime,
    obj_id: str,
    otype: str,
    lat: float,
    lon: float,
    k: int,
    producers: list[str] | None = None,
    exact: bool = False,
    partial: bool = False,
) -> list[dict]:
    """``k`` observations of one real object in the last 72 hours.

    :param producers: one producer per observation; default is random, 70% algorithm.
    :param exact: always report the specific type.
    :param partial: never report the specific type (generic or missing only).
    :return: observation rows, with ground truth in ``true_object_id``.
    """
    rows = []
    for j in range(k):
        producer = (
            producers[j]
            if producers
            else rng.choice(["algorithm", "analyst"], p=[0.7, 0.3])
        )
        olat, olon = jitter(rng, lat, lon, 80 if producer == "algorithm" else 150)
        r = rng.random()
        if exact:
            rtype = otype
        elif partial:
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
                "confidence": round(float(rng.uniform(0.4, 0.99)), 2),
                "producer": producer,
                "sensor": str(rng.choice(SENSORS)),
                "true_object_id": obj_id,
            }
        )
    return rows


def generate(now: dt.datetime, seed: int = 42) -> Sources:
    """Build the seeded synthetic sources.

    Seeded cases make the review queue real: **twin** object records close together,
    **duplicates** of one real object, **partial** observations of objects missing from the
    object system, **noise** clutter and **cross-producer** corroboration. Changing the order
    of random draws changes every downstream number.

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

    # Twins: a second record near an existing one (40 m for duplicates, 120 m for look-alikes).
    twin_src = objects.sample(N_TWIN, random_state=3).reset_index(drop=True)
    twin_pairs, twin_rows = [], []
    for i, o in twin_src.iterrows():
        is_dup = i < N_DUP
        tlat, tlon = offset(o.lat, o.lon, 40 if is_dup else 120, rng.uniform(0, 360))
        tw = {
            "object_id": f"OBJ-T{i:03d}",
            "object_type": o.object_type,
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
        lat, lon = rng.uniform(BBOX[0], BBOX[1]), rng.uniform(BBOX[2], BBOX[3])
        obs_rows += obs_of(
            f"NEW-{i:04d}", rng.choice(LEAVES), lat, lon, int(rng.integers(2, 4))
        )
    # Missing objects only ever reported with a generic type or none.
    for i in range(N_PARTIAL):
        lat, lon = rng.uniform(BBOX[0], BBOX[1]), rng.uniform(BBOX[2], BBOX[3])
        obs_rows += obs_of(
            f"NEWP-{i:04d}", rng.choice(LEAVES), lat, lon, 3, partial=True
        )
    for i, (o, tw, is_dup) in enumerate(twin_pairs):
        # Ambiguous between two nearby records; a duplicate's observations belong to the original.
        t = o if (is_dup or i % 2 == 0) else tw
        obs_rows += obs_of(t["object_id"], t["object_type"], t["lat"], t["lon"], 3)
    for _, o in objects.sample(N_CROSS, random_state=5).iterrows():
        obs_rows += obs_of(o.object_id, o.object_type, o.lat, o.lon, 2,
                           producers=["algorithm", "analyst"], exact=True)  # fmt: skip

    obs = pd.DataFrame(obs_rows)
    noise_idx = rng.choice(len(obs), int(NOISE_FRAC * len(obs)), replace=False)
    obs.loc[noise_idx, "lat"] = rng.uniform(BBOX[0], BBOX[1], len(noise_idx))
    obs.loc[noise_idx, "lon"] = rng.uniform(BBOX[2], BBOX[3], len(noise_idx))
    obs.loc[noise_idx, "reported_type"] = rng.choice(LEAVES, len(noise_idx))
    obs.loc[noise_idx, "true_object_id"] = "NOISE"
    obs["batch"] = rng.integers(1, 3, len(obs))  # delivered in two files

    # Activity history: objects that are still being observed were seen more recently.
    # A separate generator, so adding this left the observation stream unchanged.
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
        f"missing type={obs.reported_type.isna().sum()}, noise={len(noise_idx)})"
    )

    # Reports are entered after collection, never after now: a delay that would overshoot is
    # redrawn uniformly within the time left. A separate generator keeps the stream above unchanged.
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
                f"observed at {r.lat:.4f}, {r.lon:.4f} on {r.obs_time[:16].replace('T', ' ')}Z "
                f"with confidence {r.confidence}."
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
    """Write a DataFrame as JSON lines to a (volume) path."""
    with open(path, "w") as f:
        for r in df.to_dict("records"):
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
