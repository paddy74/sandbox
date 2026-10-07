# Databricks notebook source
# MAGIC %md
# MAGIC # Object resolution pipeline: observations to objects (synthetic)
# MAGIC
# MAGIC Run top to bottom on **Databricks Free Edition (serverless)**. Safe to re-run: the reset cell clears the landing
# MAGIC folders and bronze tables, and every other table is `CREATE OR REPLACE`. Re-run the whole notebook, not single cells
# MAGIC (the write-back `MERGE`s are not idempotent on their own).
# MAGIC
# MAGIC **What it does, in layer order**
# MAGIC 1. **Governed data foundation:** four mocked source types (object system, tasking CSV, observation JSON, report text)
# MAGIC    land with `COPY INTO` plus `source_file` / `ingested_at`; the public CCO ontology is flattened to a Delta table
# MAGIC    and used for type compatibility.
# MAGIC 2. **Predictive ML:** H3 blocking, candidate pairs, a scikit-learn gradient boosting match model logged to MLflow and
# MAGIC    registered in Unity Catalog. Probability drives `AUTO` (>= 0.9), `REVIEW` (0.5 to 0.9) or `NOMINATE` (< 0.5 or no
# MAGIC    candidate); nominated observations are clustered by label propagation in Spark SQL.
# MAGIC 3. **GenAI:** a sourced object dossier with `ai_query`, with every citation checked against the retrieved reports.
# MAGIC 4. **Presentation:** views for the AI/BI dashboard and Genie, plus a review queue with analyst write-back.
# MAGIC
# MAGIC > All data is synthetic. `true_object_id` is ground truth for scoring the demo only; it is never a model feature and
# MAGIC > never appears in the dossier prompt. Model numbers printed here are **in-sample** unless labelled held-out.

# COMMAND ----------

# MAGIC %pip install -q rdflib folium

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
NOTES: dict = {}  # cheat-card facts, printed in the last cell

CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "obj_resolution_demo"
S = f"{CATALOG}.{SCHEMA}"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
LANDING = {
    "obs": f"{VOL}/observations",
    "tasking": f"{VOL}/tasking",
    "reports": f"{VOL}/reports",
}
AUTO_T, REVIEW_T = 0.9, 0.5
AI_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
H3_RES = 8
MAX_DIST_M = 500
print("Using", S, "| landing:", VOL)


def hav_sql(lat1: str, lon1: str, lat2: str, lon2: str) -> str:
    """Haversine distance in metres as a Spark SQL expression."""
    return (
        f"2*6371000*asin(sqrt(pow(sin(radians({lat1}-{lat2})/2),2)"
        f" + cos(radians({lat1}))*cos(radians({lat2}))*pow(sin(radians({lon1}-{lon2})/2),2)))"
    )


# COMMAND ----------

# MAGIC %md ## 1. Schema, volume and reset

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {S}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {S}.landing")
dbutils.fs.mkdirs(f"{VOL}/ontology")

for stream in spark.streams.active:  # stop any orphaned stream from an earlier run
    stream.stop()
for path in LANDING.values():
    dbutils.fs.rm(path, True)
    dbutils.fs.mkdirs(path)
for tbl in [
    "bronze_observations",
    "bronze_tasking",
    "bronze_reports",
    "review_decisions",
]:
    spark.sql(f"DROP TABLE IF EXISTS {S}.{tbl}")

# COMMAND ----------

# MAGIC %md ## 2. Synthetic sources (Layer 1)
# MAGIC Types are **CCO-native labels**; generic "reported" types are CCO ancestors, resolved to IRIs in section 5.
# MAGIC Seeded cases so the review queue is real: **twin** OMS records close together, **duplicates** of one real object,
# MAGIC **partial** observations of objects not in the OMS, **noise** clutter and **cross-producer** corroboration.

# COMMAND ----------

rng = np.random.default_rng(42)
NOW = dt.datetime.now(dt.UTC).replace(tzinfo=None, microsecond=0)

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


def jitter(lat: float, lon: float, sigma_m: float) -> tuple[float, float]:
    """Gaussian position noise in metres."""
    dlat = rng.normal(0, sigma_m) / 111_320
    dlon = rng.normal(0, sigma_m) / (111_320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def offset(
    lat: float, lon: float, dist_m: float, bearing_deg: float
) -> tuple[float, float]:
    """Point dist_m metres from (lat, lon) on a bearing."""
    b = math.radians(bearing_deg)
    return (
        lat + dist_m * math.cos(b) / 111_320,
        lon + dist_m * math.sin(b) / (111_320 * math.cos(math.radians(lat))),
    )


# Human-readable designator: CCO Arbitrary Identifier, linked by "designates" and stored as its text value.
DESIGNATOR_IRI = "https://www.commoncoreontologies.org/ont00000923"
PHONETIC = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India",
            "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
            "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray", "Yankee", "Zulu"]  # fmt: skip


def designator(i: int) -> str:
    """Unique call-sign style designator for the i-th object, e.g. Bravo-12."""
    return f"{PHONETIC[i % len(PHONETIC)]}-{i // len(PHONETIC) + 1:02d}"


N_OBJ, N_OBSERVED, N_NEW = 2000, 1500, 150
N_TWIN, N_DUP = (
    25,
    5,
)  # twin pairs; the first N_DUP are true duplicates of one real object
N_PARTIAL, N_CROSS = 30, 100
NOISE_FRAC = 0.05
N_TASKS = 40
BBOX = (39.0, 39.5, -105.5, -105.0)  # arbitrary synthetic area

objects = pd.DataFrame(
    {
        "object_id": [f"OBJ-{i:05d}" for i in range(N_OBJ)],
        "object_type": rng.choice(LEAVES, N_OBJ),
        "lat": rng.uniform(BBOX[0], BBOX[1], N_OBJ),
        "lon": rng.uniform(BBOX[2], BBOX[3], N_OBJ),
        "marking": rng.choice(
            ["OPEN", "RESTRICTED"], N_OBJ, p=[0.8, 0.2]
        ),  # synthetic markings
        "dup_of": None,
    }
)

# Twins: a second OMS record near an existing one (40 m for duplicates, 120 m for look-alikes).
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
objects["first_seen"] = NOW - dt.timedelta(days=30)
objects["last_seen"] = NOW - dt.timedelta(days=3)
objects["obs_count"] = 0
objects["source"] = "OMS"
objects["status"] = "ACTIVE"
# Deterministic (no rng call), so the seeded random stream and the cheat-card numbers do not change.
objects["designator"] = [designator(i) for i in range(len(objects))]
objects["designator_type"] = DESIGNATOR_IRI
assert objects.designator.is_unique, "designators must be unique"


def make_obs(obj_id, otype, lat, lon, k, producers=None, exact=False, partial=False):
    """k synthetic observations of one real object; ground truth kept in true_object_id."""
    rows = []
    for j in range(k):
        producer = (
            producers[j]
            if producers
            else rng.choice(["algorithm", "analyst"], p=[0.7, 0.3])
        )
        olat, olon = jitter(lat, lon, 80 if producer == "algorithm" else 150)
        r = rng.random()
        if exact:
            rtype = otype
        elif partial:
            rtype = None if r < 0.5 else GENERIC[otype]  # never the specific type
        else:
            rtype = otype if r < 0.7 else (GENERIC[otype] if r < 0.9 else None)
        rows.append(
            {
                "obs_id": f"OBS-{rng.integers(1e12):012d}",
                "obs_time": (
                    NOW - dt.timedelta(hours=float(rng.uniform(0, 72)))
                ).isoformat(),
                "lat": olat,
                "lon": olon,
                "reported_type": rtype,
                "confidence": round(float(rng.uniform(0.4, 0.99)), 2),
                "producer": producer,
                "sensor": str(rng.choice(["EO", "SAR", "FMV"])),
                "true_object_id": obj_id,
            }
        )
    return rows


obs_rows = []
for _, o in objects.sample(N_OBSERVED, random_state=1).iterrows():  # known OMS objects
    obs_rows += make_obs(
        o.object_id, o.object_type, o.lat, o.lon, int(rng.integers(1, 5))
    )
for i in range(N_NEW):  # new objects not in the OMS: should become nominations
    lat, lon = rng.uniform(BBOX[0], BBOX[1]), rng.uniform(BBOX[2], BBOX[3])
    obs_rows += make_obs(
        f"NEW-{i:04d}", rng.choice(LEAVES), lat, lon, int(rng.integers(2, 4))
    )
for i in range(N_PARTIAL):  # new objects seen only as generic or missing type
    lat, lon = rng.uniform(BBOX[0], BBOX[1]), rng.uniform(BBOX[2], BBOX[3])
    obs_rows += make_obs(f"NEWP-{i:04d}", rng.choice(LEAVES), lat, lon, 3, partial=True)
for i, (o, tw, is_dup) in enumerate(
    twin_pairs
):  # ambiguous between two nearby OMS records
    t = (
        o if (is_dup or i % 2 == 0) else tw
    )  # duplicates always resolve to the original record
    obs_rows += make_obs(t["object_id"], t["object_type"], t["lat"], t["lon"], 3)
for _, o in objects.sample(
    N_CROSS, random_state=5
).iterrows():  # algorithm + analyst, exact type
    obs_rows += make_obs(
        o.object_id,
        o.object_type,
        o.lat,
        o.lon,
        2,
        producers=["algorithm", "analyst"],
        exact=True,
    )

obs = pd.DataFrame(obs_rows)
noise_idx = rng.choice(
    len(obs), int(NOISE_FRAC * len(obs)), replace=False
)  # clutter, no real object
obs.loc[noise_idx, "lat"] = rng.uniform(BBOX[0], BBOX[1], len(noise_idx))
obs.loc[noise_idx, "lon"] = rng.uniform(BBOX[2], BBOX[3], len(noise_idx))
obs.loc[noise_idx, "reported_type"] = rng.choice(LEAVES, len(noise_idx))
obs.loc[noise_idx, "true_object_id"] = "NOISE"
obs["batch"] = rng.integers(1, 3, len(obs))
print(
    f"objects={len(objects)} (twins={N_TWIN}, duplicates={N_DUP}) observations={len(obs)} "
    f"(generic type={obs.reported_type.isin(GENERIC_LABELS).sum()}, "
    f"missing type={obs.reported_type.isna().sum()}, noise={len(noise_idx)})"
)
NOTES["observations"] = len(obs)

# Report text: templated synthetic narrative standing in for finished reports (one per observation).
obs["report_id"] = "RPT-" + obs.obs_id.str[4:]
reports = pd.DataFrame(
    {
        "report_id": obs.report_id,
        "obs_id": obs.obs_id,
        "report_time": obs.obs_time,
        "report_text": [
            f"{r.producer.title()} report ({r.sensor}): {r.reported_type or 'unidentified object'} "
            f"observed at {r.lat:.4f}, {r.lon:.4f} on {r.obs_time[:16].replace('T', ' ')}Z "
            f"with confidence {r.confidence}."
            for r in obs.itertuples()
        ],
    }
)

# Tasking records (mock MySQL collection/tasking export).
tasking = pd.DataFrame(
    {
        "task_id": [f"TASK-{i:03d}" for i in range(N_TASKS)],
        "sensor": rng.choice(["EO", "SAR", "FMV"], N_TASKS),
        "min_lat": (la := rng.uniform(BBOX[0], BBOX[1] - 0.05, N_TASKS)),
        "max_lat": la + 0.05,
        "min_lon": (lo := rng.uniform(BBOX[2], BBOX[3] - 0.05, N_TASKS)),
        "max_lon": lo + 0.05,
        "start_time": [
            (NOW - dt.timedelta(hours=float(h))).isoformat()
            for h in rng.uniform(24, 72, N_TASKS)
        ],
        "end_time": [
            (NOW - dt.timedelta(hours=float(h))).isoformat()
            for h in rng.uniform(0, 24, N_TASKS)
        ],
        "requested_by": rng.choice(
            ["ANALYSIS-TEAM", "COMMAND-A", "COMMAND-B"], N_TASKS
        ),
        "priority": rng.integers(1, 4, N_TASKS),
    }
)

spark.createDataFrame(objects).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{S}.oms_objects")
spark.createDataFrame(
    [(o["object_id"], tw["object_id"], bool(d)) for o, tw, d in twin_pairs],
    "original_id STRING, twin_id STRING, is_duplicate BOOLEAN",
).write.mode("overwrite").saveAsTable(
    f"{S}.seed_twins"
)  # demo seeding metadata (ground truth), not a feature
NOTES["oms_objects_seeded"] = len(objects)


def write_jsonl(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame as JSON lines to a volume path."""
    with open(path, "w") as f:
        for r in df.to_dict("records"):
            f.write(json.dumps(r) + "\n")


write_jsonl(
    obs[obs.batch == 1].drop(columns=["batch", "report_id"]),
    f"{LANDING['obs']}/batch_1.json",
)
write_jsonl(reports, f"{LANDING['reports']}/reports_1.json")
tasking.to_csv(f"{LANDING['tasking']}/tasking_1.csv", index=False)

# COMMAND ----------

# MAGIC %md ## 3. Ingest with `COPY INTO` (incremental, idempotent)
# MAGIC Already-loaded files are skipped, so a re-run loads nothing new. No streaming query, checkpoint or long-running compute.

# COMMAND ----------

BRONZE_OBS = f"{S}.bronze_observations"


def ingest_observations() -> None:
    """COPY INTO bronze_observations from the landing folder; prints rows loaded."""
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {BRONZE_OBS} (
        obs_id STRING, obs_time TIMESTAMP, lat DOUBLE, lon DOUBLE, reported_type STRING,
        confidence DOUBLE, producer STRING, sensor STRING, true_object_id STRING,
        source_file STRING, ingested_at TIMESTAMP)""")
    r = spark.sql(f"""
      COPY INTO {BRONZE_OBS}
      FROM (SELECT obs_id, CAST(obs_time AS TIMESTAMP) AS obs_time,
                   CAST(lat AS DOUBLE) AS lat, CAST(lon AS DOUBLE) AS lon, reported_type,
                   CAST(confidence AS DOUBLE) AS confidence, producer, sensor, true_object_id,
                   _metadata.file_name AS source_file, current_timestamp() AS ingested_at
            FROM '{LANDING["obs"]}')
      FILEFORMAT = JSON""").first()
    print("COPY INTO bronze_observations:", r.asDict() if r else None)


ingest_observations()
n1 = spark.table(BRONZE_OBS).count()
print("rows after batch 1:", n1)

write_jsonl(
    obs[obs.batch == 2].drop(columns=["batch", "report_id"]),
    f"{LANDING['obs']}/batch_2.json",
)
ingest_observations()
n2 = spark.table(BRONZE_OBS).count()
print("rows after batch 2 (batch 1 skipped):", n2)
assert n2 == len(obs), (
    f"expected {len(obs)}, got {n2} (duplicates mean files were reloaded)"
)

ingest_observations()
assert spark.table(BRONZE_OBS).count() == len(obs), (
    "COPY INTO re-run changed the row count"
)

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {S}.bronze_reports (
    report_id STRING, obs_id STRING, report_time TIMESTAMP, report_text STRING,
    source_file STRING, ingested_at TIMESTAMP)""")
spark.sql(f"""
  COPY INTO {S}.bronze_reports
  FROM (SELECT report_id, obs_id, CAST(report_time AS TIMESTAMP) AS report_time, report_text,
               _metadata.file_name AS source_file, current_timestamp() AS ingested_at
        FROM '{LANDING["reports"]}')
  FILEFORMAT = JSON""")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {S}.bronze_tasking (
    task_id STRING, sensor STRING, min_lat DOUBLE, max_lat DOUBLE, min_lon DOUBLE, max_lon DOUBLE,
    start_time TIMESTAMP, end_time TIMESTAMP, requested_by STRING, priority INT,
    source_file STRING, ingested_at TIMESTAMP)""")
spark.sql(f"""
  COPY INTO {S}.bronze_tasking
  FROM (SELECT task_id, sensor, CAST(min_lat AS DOUBLE) AS min_lat, CAST(max_lat AS DOUBLE) AS max_lat,
               CAST(min_lon AS DOUBLE) AS min_lon, CAST(max_lon AS DOUBLE) AS max_lon,
               CAST(start_time AS TIMESTAMP) AS start_time, CAST(end_time AS TIMESTAMP) AS end_time,
               requested_by, CAST(priority AS INT) AS priority,
               _metadata.file_name AS source_file, current_timestamp() AS ingested_at
        FROM '{LANDING["tasking"]}')
  FILEFORMAT = CSV FORMAT_OPTIONS ('header' = 'true')""")
for t in ["bronze_observations", "bronze_reports", "bronze_tasking"]:
    print(t, spark.table(f"{S}.{t}").count(), "rows")

# COMMAND ----------

# MAGIC %md ## 4. H3 sanity check

# COMMAND ----------

ring = (
    spark.sql("""
    SELECT size(h3_kring(h3_longlatash3(-105.25, 39.25, 8), 1)) AS ring_size""")
    .first()
    .ring_size
)
assert ring == 7, f"unexpected H3 ring size {ring}"

# COMMAND ----------

# MAGIC %md ## 5. CCO ontology: parse, flatten, resolve demo types to IRIs
# MAGIC `type_ancestors` comes from `cco_classes.ancestor_iris`, so type compatibility is the ontology's, not a hand-built table.

# COMMAND ----------

CCO_FILE = "CommonCoreOntologiesMerged.ttl"

_cwd = Path.cwd()
CCO_CANDIDATES = [
    _cwd.parent / "data" / "ontology" / CCO_FILE,  # git folder: repo/data/ontology/
    _cwd / "data" / "ontology" / CCO_FILE,
    Path(VOL) / "ontology" / CCO_FILE,  # uploaded to the volume
]
cco_local = next(
    (p for p in CCO_CANDIDATES if p.exists() and p.stat().st_size > 500_000), None
)
if cco_local is None:
    raise FileNotFoundError(f"No valid {CCO_FILE} (>500 KB) in: {CCO_CANDIDATES}")

from rdflib import OWL, RDF, RDFS, Graph, URIRef  # noqa: E402

g = Graph()
g.parse(cco_local, format="turtle")
classes = {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
parents = {
    c: [p for p in g.objects(c, RDFS.subClassOf) if isinstance(p, URIRef)]
    for c in classes
}
memo: dict = {}


def ancestors(c, seen=()):
    """All superclasses of c (transitive), guarding against cycles."""
    if c in memo:
        return memo[c]
    out = set()
    for p in parents.get(c, []):
        if p in seen:
            continue
        out |= {p} | ancestors(p, seen + (c,))
    memo[c] = out
    return out


rows = [
    (
        str(c),
        str(g.value(c, RDFS.label) or ""),
        [str(p) for p in parents[c]],
        sorted(str(a) for a in ancestors(c)),
    )
    for c in classes
]
spark.createDataFrame(
    rows,
    "iri STRING, label STRING, parent_iris ARRAY<STRING>, ancestor_iris ARRAY<STRING>",
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{S}.cco_classes"
)
print(f"triples={len(g)} classes={len(classes)}")

labels = LEAVES + GENERIC_LABELS
spark.createDataFrame(
    [(label,) for label in labels], "label STRING"
).createOrReplaceTempView("demo_labels")
spark.sql(f"""
  CREATE OR REPLACE TABLE {S}.type_iri AS
  SELECT d.label, c.iri FROM demo_labels d JOIN {S}.cco_classes c ON lower(c.label) = lower(d.label)""")
cnt = spark.sql(f"SELECT label, count(*) n FROM {S}.type_iri GROUP BY label").toPandas()
missing = sorted(set(labels) - set(cnt.label))
ambiguous = cnt[cnt.n > 1].label.tolist()
assert not missing and not ambiguous, (
    f"CCO label problems: missing={missing} ambiguous={ambiguous}"
)
spark.sql(f"""
  CREATE OR REPLACE TABLE {S}.type_ancestors AS
  SELECT t.label AS type, a.label AS ancestor
  FROM {S}.type_iri t
  JOIN {S}.cco_classes c ON c.iri = t.iri
  JOIN {S}.type_iri a ON array_contains(c.ancestor_iris, a.iri)""")
ta = spark.table(f"{S}.type_ancestors").toPandas()
bad = [
    (leaf, gen)
    for leaf, gen in GENERIC.items()
    if (leaf, gen) not in set(zip(ta.type, ta.ancestor, strict=True))
]
assert not bad, (
    f"CCO does not place these under the generic label used by the generator: {bad}"
)
assert str(g.value(URIRef(DESIGNATOR_IRI), RDFS.label)) == "Arbitrary Identifier", (
    f"{DESIGNATOR_IRI} is not CCO Arbitrary Identifier in {cco_local}"
)
display(spark.sql(f"SELECT label, iri FROM {S}.type_iri ORDER BY label"))

# COMMAND ----------

# MAGIC %md ## 6. Candidate pairs and match model (Layer 2)
# MAGIC H3 blocking (resolution 8, ring 1, <= 500 m). Features are cast to **DOUBLE** (Spark `CASE` returns DECIMAL, which
# MAGIC breaks MLflow's JSON input example). Labels simulate past analyst adjudications via `true_object_id`.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {S}.silver_candidate_pairs AS
WITH o AS (
  SELECT *, explode(h3_kring(h3_longlatash3(lon, lat, {H3_RES}), 1)) AS cell FROM {S}.bronze_observations),
b AS (SELECT *, h3_longlatash3(lon, lat, {H3_RES}) AS cell FROM {S}.oms_objects WHERE source = 'OMS')
SELECT * FROM (
  SELECT o.obs_id, b.object_id,
    {hav_sql("o.lat", "o.lon", "b.lat", "b.lon")} AS dist_m,
    CAST(CASE WHEN o.reported_type IS NULL THEN 0.5
              WHEN o.reported_type = b.object_type THEN 1.0
              WHEN ta.ancestor IS NOT NULL THEN 0.8 ELSE 0.0 END AS DOUBLE) AS type_score,
    CAST(CAST(o.reported_type IS NULL AS INT) AS DOUBLE) AS type_missing,
    CAST(o.confidence AS DOUBLE) AS confidence,
    CAST(CAST(o.producer = 'analyst' AS INT) AS DOUBLE) AS is_analyst,
    CAST(o.true_object_id = b.object_id AS INT) AS label
  FROM o JOIN b ON o.cell = b.cell
  LEFT JOIN {S}.type_ancestors ta ON ta.type = b.object_type AND ta.ancestor = o.reported_type)
WHERE type_score > 0 AND dist_m <= {MAX_DIST_M}""")
display(
    spark.sql(
        f"SELECT label, count(*) n FROM {S}.silver_candidate_pairs GROUP BY label"
    )
)

# COMMAND ----------

import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

FEATURES = ["dist_m", "type_score", "type_missing", "confidence", "is_analyst"]

pdf = spark.table(f"{S}.silver_candidate_pairs").toPandas()
pdf[FEATURES] = pdf[FEATURES].astype("float64")  # no Decimal/object columns
X, y = pdf[FEATURES], pdf["label"]
assert y.nunique() == 2, "candidate pairs need both match and non-match labels"
train_idx, test_idx = next(
    GroupShuffleSplit(test_size=0.25, random_state=7).split(X, y, groups=pdf["obs_id"])
)
clf = HistGradientBoostingClassifier(max_iter=200, random_state=7).fit(
    X.iloc[train_idx], y.iloc[train_idx]
)
p_test = clf.predict_proba(X.iloc[test_idx])[:, 1]
metrics = {
    "held_out_auc": float(roc_auc_score(y.iloc[test_idx], p_test)),
    "held_out_avg_precision": float(average_precision_score(y.iloc[test_idx], p_test)),
}
print("HELD-OUT (synthetic, grouped by observation):", metrics)
NOTES.update({k: round(v, 4) for k, v in metrics.items()})

signature = infer_signature(X.iloc[:5], clf.predict(X.iloc[:5]))
with mlflow.start_run(run_name="obj_resolution_match_model") as run:
    mlflow.log_params(
        {"model": "HistGradientBoostingClassifier", "features": ",".join(FEATURES)}
    )
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        clf, "model", signature=signature, input_example=X.iloc[:5]
    )
RUN_ID = run.info.run_id
mlflow.set_registry_uri("databricks-uc")
mv = mlflow.register_model(f"runs:/{RUN_ID}/model", f"{S}.obj_resolution_match_model")
NOTES["uc_model_version"] = f"{S}.obj_resolution_match_model v{mv.version}"
print("registered:", NOTES["uc_model_version"])

# COMMAND ----------

# MAGIC %md ### Score every pair and apply the three-way decision
# MAGIC `AUTO` >= 0.9, `REVIEW` 0.5 to 0.9, `NOMINATE` < 0.5 or no candidate. Thresholds are demo values, not tuned.

# COMMAND ----------

pdf["match_prob"] = clf.predict_proba(pdf[FEATURES])[:, 1]
spark.createDataFrame(pdf[["obs_id", "object_id", "match_prob"] + FEATURES]).write.mode(
    "overwrite"
).option("overwriteSchema", "true").saveAsTable(f"{S}.silver_scored_pairs")
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.silver_model_decisions AS
WITH best AS (
  SELECT *, row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk FROM {S}.silver_scored_pairs)
SELECT ob.obs_id, ob.true_object_id, b.object_id AS matched_object_id, round(b.match_prob, 3) AS match_prob,
       CASE WHEN b.match_prob >= {AUTO_T} THEN 'AUTO'
            WHEN b.match_prob >= {REVIEW_T} THEN 'REVIEW'
            ELSE 'NOMINATE' END AS decision
FROM {S}.bronze_observations ob LEFT JOIN best b ON ob.obs_id = b.obs_id AND b.rk = 1""")

bands = spark.sql(f"""
  SELECT decision, count(*) AS n,
    round(avg(CASE WHEN decision = 'NOMINATE' THEN CAST(true_object_id NOT LIKE 'OBJ-%' AS DOUBLE)
                   ELSE CAST(matched_object_id = true_object_id AS DOUBLE) END), 3) AS precision_vs_synthetic_truth
  FROM {S}.silver_model_decisions GROUP BY decision""").toPandas()
bands["pct"] = (100 * bands.n / bands.n.sum()).round(1)
print("IN-SAMPLE band counts and precision (synthetic ground truth):")
print(bands.sort_values("decision").to_string(index=False))
display(spark.createDataFrame(bands.sort_values("decision")))
NOTES["bands"] = json.loads(bands.set_index("decision").to_json(orient="index"))
n_review = int(bands.loc[bands.decision == "REVIEW", "n"].sum())
print(
    f"REVIEW queue: {n_review} rows ({100 * n_review / len(obs):.1f}% of observations); present it as a ranked queue, do not tune to shrink it"
)

# Duplicate-object evidence: does the model surface the seeded duplicate OMS records as ambiguous?
flagged = spark.sql(f"""
  WITH r AS (SELECT obs_id, object_id, match_prob,
                    row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk FROM {S}.silver_scored_pairs),
  t AS (SELECT least(a.object_id, b.object_id) o1, greatest(a.object_id, b.object_id) o2, a.obs_id
        FROM r a JOIN r b ON a.obs_id = b.obs_id AND a.rk = 1 AND b.rk = 2
        WHERE b.match_prob >= 0.3 AND a.match_prob - b.match_prob <= 0.2)
  SELECT o1, o2, count(*) n_obs FROM t GROUP BY o1, o2 HAVING count(*) >= 2""").toPandas()
seeded = spark.sql(
    f"SELECT dup_of, object_id FROM {S}.oms_objects WHERE dup_of IS NOT NULL"
).toPandas()
seeded_pairs = {
    tuple(sorted(pair)) for pair in zip(seeded.dup_of, seeded.object_id, strict=True)
}
flagged_pairs = {
    tuple(sorted(pair)) for pair in zip(flagged.o1, flagged.o2, strict=True)
}
hit = len(seeded_pairs & flagged_pairs)
NOTES["duplicates_recovered"] = f"{hit}/{len(seeded_pairs)}"
print("seeded duplicate objects surfaced as ambiguous:", NOTES["duplicates_recovered"])

# COMMAND ----------

# MAGIC %md ## 7. Cluster unmatched observations into nominations
# MAGIC Connected components by iterative min-label propagation in Spark SQL (no GraphFrames). Convergence is asserted.

# COMMAND ----------

MAX_ITER = 14

spark.sql(f"""
CREATE OR REPLACE TABLE {S}.nom_obs AS
SELECT b.obs_id, b.lat, b.lon, b.reported_type
FROM {S}.silver_model_decisions d JOIN {S}.bronze_observations b ON d.obs_id = b.obs_id
WHERE d.decision = 'NOMINATE'""")
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.nom_edges AS
WITH n AS (SELECT * FROM {S}.nom_obs),
a AS (SELECT *, explode(h3_kring(h3_longlatash3(lon, lat, {H3_RES}), 1)) AS cell FROM n),
c AS (SELECT *, h3_longlatash3(lon, lat, {H3_RES}) AS cell FROM n)
SELECT a.obs_id AS src, c.obs_id AS dst
FROM a JOIN c ON a.cell = c.cell AND a.obs_id <> c.obs_id
WHERE {hav_sql("a.lat", "a.lon", "c.lat", "c.lon")} <= 400
  AND (a.reported_type IS NULL OR c.reported_type IS NULL OR a.reported_type = c.reported_type)""")
spark.sql(
    f"CREATE OR REPLACE TABLE {S}.cc_0 AS SELECT obs_id AS id, obs_id AS label FROM {S}.nom_obs"
)

converged, cur = False, f"{S}.cc_0"
for i in range(1, MAX_ITER + 1):
    prev, cur = f"{S}.cc_{(i - 1) % 2}", f"{S}.cc_{i % 2}"
    spark.sql(f"""
      CREATE OR REPLACE TABLE {cur} AS
      SELECT id, min(label) AS label FROM (
        SELECT id, label FROM {prev}
        UNION ALL
        SELECT e.dst AS id, l.label FROM {prev} l JOIN {S}.nom_edges e ON l.id = e.src) t
      GROUP BY id""")
    changed = (
        spark.sql(
            f"SELECT count(*) c FROM {cur} x JOIN {prev} y USING (id) WHERE x.label <> y.label"
        )
        .first()
        .c
    )
    print(f"iteration {i}: {changed} labels changed")
    if changed == 0:
        converged = True
        break
assert converged, (
    f"label propagation did not converge in {MAX_ITER} iterations; raise MAX_ITER"
)

spark.sql(f"CREATE OR REPLACE TABLE {S}.cc_final AS SELECT * FROM {cur}")
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.silver_nom_members AS
SELECT l.id AS obs_id, concat('NOM-', substr(sha2(l.label, 256), 1, 10)) AS object_id
FROM {S}.cc_final l
WHERE l.label IN (SELECT label FROM {S}.cc_final GROUP BY label HAVING count(*) >= 2)""")
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_nominations AS
SELECT *, concat('New-', lpad(CAST(row_number() OVER (ORDER BY object_id) AS STRING), 3, '0')) AS designator,
       '{DESIGNATOR_IRI}' AS designator_type
FROM (SELECT m.object_id,
             coalesce(max(c.reported_type), 'Unknown') AS object_type,
             avg(c.lat) AS lat, avg(c.lon) AS lon, count(*) AS obs_count
      FROM {S}.silver_nom_members m JOIN {S}.nom_obs c ON m.obs_id = c.obs_id
      GROUP BY m.object_id)""")
r = spark.sql(
    f"SELECT count(*) noms, coalesce(sum(obs_count), 0) recovered FROM {S}.gold_nominations"
).first()
NOTES["nominations"], NOTES["observations_recovered"] = r.noms, int(r.recovered)
NOTES["recovered_share_of_observations_pct"] = round(100 * r.recovered / len(obs), 1)
print(
    f"nominations={r.noms}, observations recovered (would otherwise be discarded)={r.recovered}"
)

# COMMAND ----------

# MAGIC %md ## 8. Write-back to the object system (`MERGE`)
# MAGIC `AUTO` associations update the object; nominations insert new objects. Re-run the whole notebook, not this cell alone.

# COMMAND ----------

NOTES["objects_before_merge"] = spark.table(f"{S}.oms_objects").count()
spark.sql(f"""
MERGE INTO {S}.oms_objects t
USING (SELECT matched_object_id AS object_id, count(*) AS n
       FROM {S}.silver_model_decisions WHERE decision = 'AUTO' GROUP BY matched_object_id) s
ON t.object_id = s.object_id
WHEN MATCHED THEN UPDATE SET t.obs_count = t.obs_count + s.n, t.last_seen = current_timestamp()""")
spark.sql(f"""
MERGE INTO {S}.oms_objects t
USING {S}.gold_nominations s ON t.object_id = s.object_id
WHEN NOT MATCHED THEN INSERT (object_id, object_type, lat, lon, marking, first_seen, last_seen,
                              obs_count, source, status, designator, designator_type)
VALUES (s.object_id, s.object_type, s.lat, s.lon, 'OPEN', current_timestamp(), current_timestamp(),
        s.obs_count, 'DATABRICKS_NOMINATION', 'NOMINATED', s.designator, s.designator_type)""")
NOTES["objects_after_merge"] = spark.table(f"{S}.oms_objects").count()
print(
    f"OMS objects: {NOTES['objects_before_merge']} before, {NOTES['objects_after_merge']} after nominations"
)
display(
    spark.sql(f"SELECT source, status, count(*) n FROM {S}.oms_objects GROUP BY ALL")
)

# COMMAND ----------

# MAGIC %md ## 9. Hero objects for the demo
# MAGIC **A** clean auto-associate (Armored Fighting Vehicle, corroborated) · **B** twin-object review case ·
# MAGIC **C** nominated cluster of individually weak observations. Selected from the data, then stored in `demo_heroes`.

# COMMAND ----------

weak_list = ", ".join(f"'{t}'" for t in GENERIC_LABELS)

hero_a = spark.sql(f"""
  SELECT d.matched_object_id AS object_id, count(*) n_obs, count(DISTINCT b.producer) n_producers,
         round(avg(d.match_prob), 3) AS avg_prob
  FROM {S}.silver_model_decisions d
  JOIN {S}.bronze_observations b ON d.obs_id = b.obs_id
  JOIN {S}.oms_objects o ON o.object_id = d.matched_object_id
  WHERE d.decision = 'AUTO' AND o.object_type = 'Armored Fighting Vehicle' AND o.source = 'OMS'
  GROUP BY d.matched_object_id HAVING count(*) >= 3
  ORDER BY n_producers DESC, n_obs DESC, avg_prob DESC LIMIT 1""").first()
assert hero_a is not None, "no Armored Fighting Vehicle with >= 3 AUTO observations"

hero_b = spark.sql(f"""
  WITH r AS (SELECT obs_id, object_id, match_prob,
                    row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk FROM {S}.silver_scored_pairs)
  SELECT a.obs_id, a.object_id AS object_id, b.object_id AS alt_object_id,
         round(a.match_prob, 3) AS p1, round(b.match_prob, 3) AS p2, x.object_type
  FROM r a JOIN r b ON a.obs_id = b.obs_id AND a.rk = 1 AND b.rk = 2
  JOIN {S}.oms_objects x ON x.object_id = a.object_id
  JOIN {S}.silver_model_decisions d ON d.obs_id = a.obs_id AND d.decision = 'REVIEW'
  JOIN {S}.seed_twins t ON (t.original_id = a.object_id AND t.twin_id = b.object_id)
                        OR (t.twin_id = a.object_id AND t.original_id = b.object_id)
  WHERE b.match_prob >= 0.3 AND NOT t.is_duplicate
  ORDER BY (x.object_type = 'Truck') DESC, abs(a.match_prob - 0.65) ASC LIMIT 1""").first()
assert hero_b is not None, (
    "no REVIEW observation sits between two seeded look-alike twins"
)

hero_c = spark.sql(f"""
  SELECT m.object_id, count(*) n_obs,
         sum(CASE WHEN b.reported_type IS NULL OR b.reported_type IN ({weak_list}) THEN 1 ELSE 0 END) AS weak_obs
  FROM {S}.silver_nom_members m JOIN {S}.bronze_observations b ON m.obs_id = b.obs_id
  GROUP BY m.object_id HAVING count(*) >= 3
  ORDER BY sum(CASE WHEN b.true_object_id LIKE 'NEWP-%' THEN 1 ELSE 0 END) DESC, n_obs DESC LIMIT 1""").first()
assert hero_c is not None, "no nominated cluster with >= 3 observations"

heroes = [
    ("A", hero_a.object_id, None, None,
     f"{hero_a.n_obs} AUTO observations from {hero_a.n_producers} producer(s), mean match_prob {hero_a.avg_prob}"),
    ("B", hero_b.object_id, hero_b.obs_id, hero_b.alt_object_id,
     f"{hero_b.object_type}; REVIEW obs between twins: match_prob {hero_b.p1} vs runner-up {hero_b.p2}"),
    ("C", hero_c.object_id, None, None,
     f"nominated from {hero_c.n_obs} observations, {hero_c.weak_obs} with generic or missing type"),
]  # fmt: skip
spark.createDataFrame(
    heroes,
    "hero STRING, object_id STRING, obs_id STRING, alt_object_id STRING, detail STRING",
).write.mode("overwrite").saveAsTable(f"{S}.demo_heroes")
NOTES["heroes"] = {
    h[0]: {"object_id": h[1], "obs_id": h[2], "alt": h[3], "detail": h[4]}
    for h in heroes
}
display(spark.table(f"{S}.demo_heroes"))

# COMMAND ----------

# MAGIC %md ## 10. Review queue and analyst write-back (Layer 4)
# MAGIC `review_queue` is the `REVIEW` band ranked by match probability, with the runner-up candidate and the features
# MAGIC behind the score. An analyst decision is stored in `review_decisions` and becomes a **training label** for the next
# MAGIC retrain (`training_labels_from_review`); the retrain itself is described, not run, in this demo.

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {S}.review_decisions (
    obs_id STRING, object_id STRING, decision STRING, decided_by STRING, decided_at TIMESTAMP)""")
spark.sql(f"""
CREATE OR REPLACE VIEW {S}.review_queue AS
WITH ranked AS (
  SELECT *, row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk FROM {S}.silver_scored_pairs)
SELECT d.obs_id, d.matched_object_id AS candidate_object_id, o.object_type AS candidate_type,
       d.match_prob, round(p1.dist_m, 0) AS dist_m, p1.type_score, b.reported_type, b.producer, b.sensor,
       b.confidence, p2.object_id AS runner_up_object_id, round(p2.match_prob, 3) AS runner_up_prob,
       b.lat, b.lon, b.obs_time, rd.decision AS analyst_decision
FROM {S}.silver_model_decisions d
JOIN {S}.bronze_observations b ON b.obs_id = d.obs_id
JOIN {S}.oms_objects o ON o.object_id = d.matched_object_id
JOIN ranked p1 ON p1.obs_id = d.obs_id AND p1.rk = 1
LEFT JOIN ranked p2 ON p2.obs_id = d.obs_id AND p2.rk = 2
LEFT JOIN {S}.review_decisions rd ON rd.obs_id = d.obs_id
WHERE d.decision = 'REVIEW'""")
spark.sql(f"""
CREATE OR REPLACE VIEW {S}.training_labels_from_review AS
SELECT obs_id, object_id, CAST(decision = 'APPROVE' AS INT) AS label FROM {S}.review_decisions""")


def adjudicate(
    obs_id: str, object_id: str, decision: str, decided_by: str = "analyst_demo"
) -> None:
    """Record an analyst APPROVE/REJECT for one observation; APPROVE also updates the object system."""
    if decision not in ("APPROVE", "REJECT"):
        raise ValueError(f"decision must be APPROVE or REJECT, got {decision!r}")
    if spark.sql(
        f"SELECT 1 FROM {S}.review_decisions WHERE obs_id = '{obs_id}'"
    ).count():
        print(f"{obs_id} already adjudicated; no change")
        return
    spark.sql(f"""
      INSERT INTO {S}.review_decisions
      VALUES ('{obs_id}', '{object_id}', '{decision}', '{decided_by}', current_timestamp())""")
    if decision == "APPROVE":
        spark.sql(f"""
          UPDATE {S}.oms_objects SET obs_count = obs_count + 1, last_seen = current_timestamp()
          WHERE object_id = '{object_id}'""")
    print(f"{decision}: {obs_id} -> {object_id}")


print("REVIEW queue rows:", spark.table(f"{S}.review_queue").count())
display(spark.sql(f"SELECT * FROM {S}.review_queue ORDER BY match_prob DESC LIMIT 20"))
print("Hero B before the decision:")
display(spark.sql(f"SELECT * FROM {S}.review_queue WHERE obs_id = '{hero_b.obs_id}'"))

# COMMAND ----------

# Analyst approves the model's top candidate for hero B.
before = (
    spark.sql(
        f"SELECT obs_count FROM {S}.oms_objects WHERE object_id = '{hero_b.object_id}'"
    )
    .first()
    .obs_count
)
adjudicate(hero_b.obs_id, hero_b.object_id, "APPROVE")
after = (
    spark.sql(
        f"SELECT obs_count FROM {S}.oms_objects WHERE object_id = '{hero_b.object_id}'"
    )
    .first()
    .obs_count
)
print(f"{hero_b.object_id} obs_count: {before} -> {after}")
display(spark.table(f"{S}.review_decisions"))
display(spark.table(f"{S}.training_labels_from_review"))

# COMMAND ----------

# MAGIC %md ### 10b. Analyst decision form
# MAGIC Pick a pending review item, choose APPROVE or REJECT, set **confirm = yes**, then run the cell below the form.
# MAGIC The decision is written to `review_decisions` by `adjudicate()` (an APPROVE also updates the object) and shows in
# MAGIC the dashboard's review-queue table after a refresh. `confirm` defaults to `no`, so running the whole notebook
# MAGIC never submits a decision. Re-run the form cell to refresh the list of pending items.

# COMMAND ----------

pending = spark.sql(f"""
  SELECT obs_id, candidate_object_id, match_prob
  FROM {S}.review_queue WHERE analyst_decision IS NULL
  ORDER BY match_prob DESC LIMIT 25""").collect()
assert pending, "no pending review items"
review_choices = {
    f"{r.obs_id} | {r.candidate_object_id} | p={r.match_prob}": (
        r.obs_id,
        r.candidate_object_id,
    )
    for r in pending
}
dbutils.widgets.removeAll()
dbutils.widgets.dropdown(
    "review_item",
    next(iter(review_choices)),
    list(review_choices),
    "1. Review item (obs | candidate | match_prob)",
)
dbutils.widgets.dropdown("decision", "APPROVE", ["APPROVE", "REJECT"], "2. Decision")
dbutils.widgets.dropdown("confirm", "no", ["no", "yes"], "3. Confirm (yes to submit)")
display(
    spark.createDataFrame(
        [(k, *v) for k, v in review_choices.items()],
        "choice STRING, obs_id STRING, candidate_object_id STRING",
    )
)

# COMMAND ----------

if dbutils.widgets.get("confirm") != "yes":
    print(
        "Nothing submitted. Set confirm = yes in the form above, then re-run this cell."
    )
else:
    choice = dbutils.widgets.get("review_item")
    if choice not in review_choices:
        raise ValueError(
            "The list is stale: re-run the form cell above, then choose again"
        )
    form_obs, form_object = review_choices[choice]
    adjudicate(form_obs, form_object, dbutils.widgets.get("decision"))
    dbutils.widgets.remove("confirm")
    dbutils.widgets.dropdown(
        "confirm", "no", ["no", "yes"], "3. Confirm (yes to submit)"
    )
    print("Submitted. Re-run the form cell to refresh the pending list.")
display(spark.table(f"{S}.review_decisions"))

# COMMAND ----------

# MAGIC %md ## 11. Sourced object dossier (Layer 3)
# MAGIC Retrieve an object's linked observations, match probabilities and report text from governed tables; generate a short
# MAGIC dossier with `ai_query`; **verify every cited report ID is in the retrieved set**. A failed check or endpoint error
# MAGIC falls back to a templated dossier, labelled as such.

# COMMAND ----------


def retrieve(object_id: str, limit: int = 10) -> list:
    """Reports linked to an object by candidate-pair probability (>= 0.3), strongest first."""
    return spark.sql(f"""
      SELECT r.report_id, p.obs_id, round(p.match_prob, 3) AS match_prob, b.producer, r.report_text
      FROM {S}.silver_scored_pairs p
      JOIN {S}.bronze_reports r ON r.obs_id = p.obs_id
      JOIN {S}.bronze_observations b ON b.obs_id = p.obs_id
      WHERE p.object_id = '{object_id}' AND p.match_prob >= 0.3
      ORDER BY p.match_prob DESC LIMIT {limit}""").collect()


def template_dossier(obj, rows: list) -> str:
    """No-LLM fallback: states the strongest linked report and the action."""
    top = rows[0]
    return (
        f"{obj.object_type} {obj.designator} (system ID {obj.object_id}) is in the object system with "
        f"{len(rows)} linked reports. "
        f"Strongest link: [{top.report_id}] at match probability {top.match_prob}. "
        f"Recommended action: analyst review of the linked reports."
    )


def build_dossier(object_id: str) -> tuple:
    """Return (text, cited_ids, citations_valid, generated_by) for one object."""
    obj = spark.sql(
        f"SELECT * FROM {S}.oms_objects WHERE object_id = '{object_id}'"
    ).first()
    rows = retrieve(object_id)
    assert rows, f"no linked reports for {object_id}"
    allowed = {r.report_id for r in rows}
    context = "\n".join(
        f"[{r.report_id}] ({r.producer}, match probability {r.match_prob}) {r.report_text}"
        for r in rows
    )
    prompt = (
        f"You are assisting an analyst. Object {obj.object_type} {obj.designator} (system ID {obj.object_id}) "
        f"is in the object management system. Refer to it as {obj.object_type} {obj.designator}. "
        f"Using ONLY the reports below, write a 4-sentence dossier: what the "
        f"object is, how consistently it has been observed, any uncertainty, and a recommended analyst action. "
        f"Cite report IDs in square brackets, one ID per bracket. Synthetic data.\n\nReports:\n{context}"
    )
    assert "true_object_id" not in prompt, "ground truth must not reach the prompt"
    try:
        text = (
            spark.sql(
                f"SELECT ai_query('{AI_ENDPOINT}', :prompt) AS d",
                args={"prompt": prompt},
            )
            .first()
            .d
        )
    except Exception as e:  # live-demo safety: say so, then fall back
        print(
            f"ai_query failed for {object_id} ({type(e).__name__}); using template dossier"
        )
        return (
            template_dossier(obj, rows),
            sorted(allowed)[:1],
            True,
            "template (ai_query error)",
        )
    cited = {
        c.strip() for grp in re.findall(r"\[([^\]]+)\]", text) for c in grp.split(",")
    }
    valid = bool(cited) and cited <= allowed
    if not valid:
        print(
            f"citation check FAILED for {object_id}: unknown or missing IDs {sorted(cited - allowed)}"
        )
        return (
            template_dossier(obj, rows),
            [rows[0].report_id],
            True,
            "template (citation check failed)",
        )
    return text, sorted(cited), True, AI_ENDPOINT


dossiers = []
for hero, oid in [("A", hero_a.object_id), ("B", hero_b.object_id)]:
    text, cited, valid, by = build_dossier(oid)
    dossiers.append(
        (
            oid,
            hero,
            text,
            cited,
            valid,
            by,
            dt.datetime.now(dt.UTC).replace(tzinfo=None),
        )
    )
    print(
        f"\n=== Hero {hero}: {oid} (generated by {by}; citations valid: {valid}) ===\n{text}"
    )
spark.createDataFrame(
    dossiers,
    "object_id STRING, hero STRING, dossier_text STRING, cited_ids ARRAY<STRING>, "
    "citations_valid BOOLEAN, generated_by STRING, generated_at TIMESTAMP",
).write.mode("overwrite").saveAsTable(f"{S}.object_dossiers")

# COMMAND ----------

# MAGIC %md ## 12. Governance: row filter on synthetic markings
# MAGIC Single-user workspace, so two identities cannot be shown. This demonstrates the mechanism; the filter is dropped again
# MAGIC so dashboards and Genie see all rows.

# COMMAND ----------

spark.sql(f"""
  CREATE OR REPLACE FUNCTION {S}.marking_filter(marking STRING)
  RETURN marking = 'OPEN' OR is_account_group_member('admins')""")
total = spark.table(f"{S}.oms_objects").count()
spark.sql(f"ALTER TABLE {S}.oms_objects SET ROW FILTER {S}.marking_filter ON (marking)")
print(
    f"rows visible with the filter: {spark.table(f'{S}.oms_objects').count()} of {total}"
)
spark.sql(f"ALTER TABLE {S}.oms_objects DROP ROW FILTER")

# COMMAND ----------

# MAGIC %md ## 13. Map for hero A (folium)

# COMMAND ----------

import folium  # noqa: E402

obj_a = spark.sql(
    f"SELECT * FROM {S}.oms_objects WHERE object_id = '{hero_a.object_id}'"
).first()
pts = spark.sql(f"""
  SELECT b.obs_id, b.lat, b.lon, b.reported_type, b.producer, d.match_prob
  FROM {S}.silver_model_decisions d JOIN {S}.bronze_observations b ON b.obs_id = d.obs_id
  WHERE d.matched_object_id = '{hero_a.object_id}' AND d.decision = 'AUTO'""").collect()
m = folium.Map(location=[obj_a.lat, obj_a.lon], zoom_start=15)
folium.Marker(
    [obj_a.lat, obj_a.lon],
    tooltip=f"{obj_a.object_id} ({obj_a.object_type})",
    icon=folium.Icon(color="red"),
).add_to(m)
for pt in pts:
    folium.CircleMarker(
        [pt.lat, pt.lon],
        radius=6,
        tooltip=f"{pt.obs_id} | {pt.reported_type} | {pt.producer} | p={pt.match_prob}",
    ).add_to(m)
displayHTML(m._repr_html_())

# COMMAND ----------

# MAGIC %md ## 14. Views for the AI/BI dashboard and Genie
# MAGIC Point the dashboard at `dash_map_points`, `dash_decisions`, `dash_object_summary` and `review_queue`. For Genie, add
# MAGIC `oms_objects`, `silver_model_decisions`, `gold_nominations`, `review_queue` and `object_dossiers`.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {S}.dash_decisions AS
SELECT decision, count(*) AS observations,
       round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct, round(avg(match_prob), 3) AS avg_match_prob
FROM {S}.silver_model_decisions GROUP BY decision""")
spark.sql(f"""
CREATE OR REPLACE VIEW {S}.dash_object_summary AS
SELECT object_type, source, status, count(*) AS objects, sum(obs_count) AS linked_observations
FROM {S}.oms_objects GROUP BY object_type, source, status""")
spark.sql(f"""
CREATE OR REPLACE VIEW {S}.dash_map_points AS
SELECT 'object' AS kind, object_id AS id, lat, lon, object_type AS label, status AS detail, CAST(NULL AS DOUBLE) AS match_prob
FROM {S}.oms_objects
UNION ALL
SELECT 'observation', b.obs_id, b.lat, b.lon, coalesce(b.reported_type, 'unidentified'), d.decision, d.match_prob
FROM {S}.silver_model_decisions d JOIN {S}.bronze_observations b ON b.obs_id = d.obs_id""")
display(spark.table(f"{S}.dash_decisions"))
display(spark.table(f"{S}.dash_object_summary"))

# COMMAND ----------

# MAGIC %md ## Cheat card
# MAGIC Numbers to read off during the demo. All synthetic; band precision is in-sample, AUC and average precision are held-out.

# COMMAND ----------

NOTES["elapsed_minutes"] = round((time.time() - T0) / 60, 1)
print(json.dumps(NOTES, indent=2, default=str))
display(spark.table(f"{S}.demo_heroes"))
