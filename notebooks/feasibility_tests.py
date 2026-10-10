# Databricks notebook source
# MAGIC %md
# MAGIC # Feasibility checks: can this design run on Databricks?
# MAGIC
# MAGIC Run **only in Databricks** (Free Edition serverless unless noted) **before planning or implementing** a design that
# MAGIC depends on a platform feature, package or service. Each design decision gets its own **independent** `check(...)`
# MAGIC block that records **PASS / FAIL / SKIP**, the error, and a named **fallback**; the last cell prints a summary table.
# MAGIC A FAIL does not stop the notebook, so one run answers every question.
# MAGIC
# MAGIC **Rules for a check block**
# MAGIC - It creates its own tiny inline data (a few rows) and its own `t_*` tables or `<name>/` volume folder.
# MAGIC   It never reads another check's output, so any block can be run, deleted or reordered alone.
# MAGIC - It exercises the **minimum** the design needs (install a package, run a SQL function, call an endpoint).
# MAGIC - It names a fallback in case it fails. If a block depends on an earlier step *inside the same section*
# MAGIC   (for example registering the model it just trained), it `skip(...)`s when that step failed.
# MAGIC - Only the schema/volume check and the setup cells are shared prerequisites.
# MAGIC
# MAGIC **How to use it**
# MAGIC 1. Add a block (copy the template at the end) for each platform-dependent design decision.
# MAGIC 2. Run top to bottom; read the summary. Facts worth keeping go in `NOTES` (e.g., which endpoint worked).
# MAGIC 3. Change the design or adopt the fallback for anything that failed, then record the decision in the plan.
# MAGIC
# MAGIC **Design decisions currently checked**
# MAGIC
# MAGIC | Section | Decision it de-risks |
# MAGIC |---|---|
# MAGIC | 0 | Runtime and package versions available |
# MAGIC | 1 | Unity Catalog schema and volume can be created and written |
# MAGIC | 2 | `COPY INTO` is incremental and idempotent (Auto Loader optional) |
# MAGIC | 3 | H3 SQL functions and H3-blocked proximity joins |
# MAGIC | 4 | `rdflib` installs and can parse and flatten an ontology to Delta (and, optionally, the repo's CCO file) |
# MAGIC | 5 | scikit-learn training, MLflow logging, Unity Catalog model registration, Spark `DECIMAL` handling |
# MAGIC | 6 | Iterative clustering (label propagation) in Spark SQL converges |
# MAGIC | 7 | `MERGE` write-back to a Delta table |
# MAGIC | 8 | Unity Catalog row filters |
# MAGIC | 9 | `folium` maps render via `displayHTML` |
# MAGIC | 10 | `ai_query` endpoints respond and can produce a cited summary |
# MAGIC | 11 | Genie space create, export and update through the REST API (scripted setup) |
# MAGIC | 12 | Manual UI checks (Genie, Databricks Apps, lineage, dashboards) |
# MAGIC
# MAGIC > This is a throwaway harness in its own schema (`feasibility_checks`), not part of any pipeline. All data is synthetic.
# MAGIC > Uncomment the last cell to remove everything it created.

# COMMAND ----------

# MAGIC %pip install -q rdflib folium

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import time
from contextlib import contextmanager

RESULTS = []
NOTES = {}  # facts worth recording from a run (e.g., working ai_query endpoint)


@contextmanager
def check(name, fallback=""):
    """Run a block, record PASS/FAIL without stopping the notebook."""
    t0 = time.time()
    print(f"\n=== {name} ===")
    try:
        yield
        RESULTS.append((name, "PASS", round(time.time() - t0, 1), "", fallback))
        print(f"PASS ({time.time() - t0:.1f}s)")
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:400]}"
        RESULTS.append((name, "FAIL", round(time.time() - t0, 1), msg, fallback))
        print(f"FAIL: {msg}\n  Fallback: {fallback}")


def skip(name, reason, fallback=""):
    """Record a check that could not run because a step it needs failed."""
    RESULTS.append((name, "SKIP", 0.0, reason, fallback))
    print(f"\n=== {name} ===\nSKIP: {reason}")


CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "feasibility_checks"
S = f"{CATALOG}.{SCHEMA}"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
print("Using", S, "| volume:", VOL)

# COMMAND ----------

# MAGIC %md ## 0. Runtime and package versions

# COMMAND ----------

with check(
    "Runtime and package versions",
    "Pin versions in %pip install or the app's requirements.txt",
):
    import sys

    versions = {"python": sys.version.split()[0], "spark": spark.version}
    from importlib.metadata import PackageNotFoundError, version

    # Some packages ship under an alternate distribution name (e.g., mlflow-skinny on Databricks).
    dists = {"mlflow": ["mlflow", "mlflow-skinny"]}
    for pkg in ["numpy", "pandas", "scikit-learn", "mlflow", "rdflib", "folium"]:
        versions[pkg] = "NOT INSTALLED"
        for dist in dists.get(pkg, [pkg]):
            try:
                versions[pkg] = version(dist)
                break
            except PackageNotFoundError:
                continue
    NOTES["versions"] = versions
    print(json.dumps(versions, indent=2))

# COMMAND ----------

# MAGIC %md ## 1. Unity Catalog schema and volume
# MAGIC Shared prerequisite: later checks write tables to this schema and files to this volume.

# COMMAND ----------

with check(
    "UC schema + volume + file write",
    "Use the default schema and an existing volume, or write to workspace files",
):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {S}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {S}.landing")
    dbutils.fs.put(f"{VOL}/_probe.txt", "ok", True)
    assert dbutils.fs.head(f"{VOL}/_probe.txt") == "ok"

# COMMAND ----------

# MAGIC %md ## 2. Incremental ingestion with `COPY INTO`
# MAGIC `COPY INTO` is idempotent and incremental (already-loaded files are skipped) and needs no streaming query,
# MAGIC checkpoint or long-running compute. Auto Loader is an optional check (`TRY_AUTOLOADER = True`) because it hung on
# MAGIC Free Edition serverless.

# COMMAND ----------

COPY_DIR = f"{VOL}/copy_into"
COPY_TBL = f"{S}.t_copy_into"


def _put_json(path, rows):
    """Write rows as newline-delimited JSON to a volume path."""
    dbutils.fs.put(path, "\n".join(json.dumps(r) for r in rows), True)


def _copy_into():
    spark.sql(
        f"""CREATE TABLE IF NOT EXISTS {COPY_TBL} (
              id STRING, value DOUBLE, source_file STRING, ingested_at TIMESTAMP)"""
    )
    return spark.sql(
        f"""
        COPY INTO {COPY_TBL}
        FROM (SELECT id, CAST(value AS DOUBLE) AS value,
                     _metadata.file_name AS source_file, current_timestamp() AS ingested_at
              FROM '{COPY_DIR}')
        FILEFORMAT = JSON"""
    ).first()


with check("COPY INTO batch 1", "spark.read.json on the volume + append"):
    dbutils.fs.rm(COPY_DIR, True)
    spark.sql(f"DROP TABLE IF EXISTS {COPY_TBL}")
    _put_json(
        f"{COPY_DIR}/batch_1.json",
        [{"id": "a", "value": 1.0}, {"id": "b", "value": 2.5}, {"id": "c", "value": 3}],
    )
    print("COPY INTO result:", _copy_into())
    n = spark.table(COPY_TBL).count()
    assert n == 3, f"expected 3 rows, got {n}"

with check(
    "COPY INTO incremental (batch 2 only; batch 1 skipped)",
    "Track loaded file names in a log table",
):
    _put_json(
        f"{COPY_DIR}/batch_2.json",
        [{"id": "d", "value": 4.0}, {"id": "e", "value": 5.0}],
    )
    _copy_into()
    n = spark.table(COPY_TBL).count()
    assert n == 5, f"expected 5 rows, got {n} (more means files were reloaded)"

with check("COPY INTO idempotent (re-run loads nothing)", "Same as above"):
    _copy_into()
    assert spark.table(COPY_TBL).count() == 5

# Optional: Auto Loader. Leave False unless you want to retry it; it has a hard 5-minute timeout.
TRY_AUTOLOADER = False

if TRY_AUTOLOADER:
    with check("Auto Loader (optional)", "Use COPY INTO (above)"):
        ckpt = f"{VOL}/_checkpoints/autoloader"
        dbutils.fs.rm(ckpt, True)
        spark.sql(f"DROP TABLE IF EXISTS {S}.t_autoloader")
        q = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.schemaLocation", ckpt + "_schema")
            .load(COPY_DIR)
            .writeStream.option("checkpointLocation", ckpt)
            .trigger(availableNow=True)
            .toTable(f"{S}.t_autoloader")
        )
        if not q.awaitTermination(300):
            progress = q.lastProgress
            q.stop()
            raise TimeoutError(
                f"stream did not finish in 300s; last progress: {progress}"
            )
        print("rows:", spark.table(f"{S}.t_autoloader").count())

# COMMAND ----------

# MAGIC %md ## 3. H3 geospatial SQL

# COMMAND ----------

with check(
    "H3 SQL functions",
    "Grid blocking on rounded lat/lon (e.g., 0.005 deg cells + neighbors)",
):
    r = spark.sql(
        """SELECT h3_longlatash3(-105.25, 39.25, 8) AS cell,
                  size(h3_kring(h3_longlatash3(-105.25, 39.25, 8), 1)) AS ring_size"""
    ).first()
    print(r)
    assert r.ring_size == 7

with check(
    "H3-blocked proximity join + haversine", "Rounded-grid blocking; same distance SQL"
):
    pts = [
        ("near_a", 39.2500, -105.2500),
        ("near_b", 39.2505, -105.2500),
        ("far", 39.4000, -105.1000),
    ]
    spark.createDataFrame(
        pts, "id STRING, lat DOUBLE, lon DOUBLE"
    ).createOrReplaceTempView("h3_pts")
    pairs = spark.sql(
        """
        WITH a AS (SELECT *, explode(h3_kring(h3_longlatash3(lon, lat, 8), 1)) AS cell FROM h3_pts),
             b AS (SELECT *, h3_longlatash3(lon, lat, 8) AS cell FROM h3_pts)
        SELECT a.id AS a_id, b.id AS b_id,
               2*6371000*asin(sqrt(pow(sin(radians(a.lat-b.lat)/2),2)
                 + cos(radians(a.lat))*cos(radians(b.lat))*pow(sin(radians(a.lon-b.lon)/2),2))) AS dist_m
        FROM a JOIN b ON a.cell = b.cell AND a.id < b.id"""
    ).collect()
    print([(p.a_id, p.b_id, round(p.dist_m)) for p in pairs])
    assert len(pairs) == 1 and {pairs[0].a_id, pairs[0].b_id} == {"near_a", "near_b"}
    assert 40 < pairs[0].dist_m < 70, f"unexpected distance {pairs[0].dist_m}"

# COMMAND ----------

# MAGIC %md ## 4. Ontology parsing with `rdflib`
# MAGIC The first check uses a three-class ontology inline. The optional second check parses the repo's
# MAGIC `data/ontology/CommonCoreOntologiesMerged.ttl` (or a copy uploaded to the volume) to test it at real size.

# COMMAND ----------

TTL = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/> .
ex:Vehicle a owl:Class ; rdfs:label "Vehicle" .
ex:GroundVehicle a owl:Class ; rdfs:label "Ground Vehicle" ; rdfs:subClassOf ex:Vehicle .
ex:Truck a owl:Class ; rdfs:label "Truck" ; rdfs:subClassOf ex:GroundVehicle .
"""


def flatten_classes(g):
    """Return (iri, label, parent_iris, ancestor_iris) rows for every named owl:Class in graph g."""
    from rdflib import OWL, RDF, RDFS, URIRef

    classes = {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    parents = {
        c: [p for p in g.objects(c, RDFS.subClassOf) if isinstance(p, URIRef)]
        for c in classes
    }
    memo = {}

    def anc(c, seen=()):
        if c in memo:
            return memo[c]
        out = set()
        for p in parents.get(c, []):
            if p in seen:
                continue
            out |= {p} | anc(p, seen + (c,))
        memo[c] = out
        return out

    return [
        (
            str(c),
            str(g.value(c, RDFS.label) or ""),
            [str(p) for p in parents[c]],
            sorted(str(a) for a in anc(c)),
        )
        for c in classes
    ]


ONTOLOGY_SCHEMA = (
    "iri STRING, label STRING, parent_iris ARRAY<STRING>, ancestor_iris ARRAY<STRING>"
)

with check(
    "rdflib parse + flatten to Delta (inline ontology)",
    "Hand-build the type hierarchy as a small table",
):
    from rdflib import Graph

    g = Graph()
    g.parse(data=TTL, format="turtle")
    rows = flatten_classes(g)
    spark.createDataFrame(rows, ONTOLOGY_SCHEMA).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{S}.t_ontology_inline")
    hit = (
        spark.sql(
            f"""SELECT count(*) c FROM {S}.t_ontology_inline
            WHERE label = 'Truck' AND array_contains(ancestor_iris, 'http://example.org/Vehicle')"""
        )
        .first()
        .c
    )
    assert hit == 1, "Truck should have Vehicle among its ancestors"

CCO_FILE = "CommonCoreOntologiesMerged.ttl"
CCO_CANDIDATES = [
    os.path.join(
        os.getcwd(), "..", "data", "ontology", CCO_FILE
    ),  # git folder: repo/data/ontology/
    os.path.join(os.getcwd(), "data", "ontology", CCO_FILE),
    f"{VOL}/ontology/{CCO_FILE}",  # uploaded to the volume
]
cco_local = next(
    (os.path.abspath(p) for p in CCO_CANDIDATES if os.path.exists(p)), None
)

if cco_local:
    with check(
        "rdflib parse + flatten the repo's CCO file (optional)",
        "Skip the full ontology; use a hand-built hierarchy",
    ):
        from rdflib import Graph

        size = os.path.getsize(cco_local)
        assert size > 500_000, (
            f"{cco_local} is only {size} bytes; likely an imports-only stub"
        )
        g = Graph()
        g.parse(cco_local, format="turtle")
        rows = flatten_classes(g)
        spark.createDataFrame(rows, ONTOLOGY_SCHEMA).write.mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(f"{S}.t_ontology_cco")
        print(
            f"{cco_local} ({size / 1e6:.1f} MB): triples={len(g)} classes={len(rows)}"
        )
        assert len(rows) > 1000, f"only {len(rows)} classes parsed"
else:
    skip(
        "rdflib parse + flatten the repo's CCO file (optional)",
        f"{CCO_FILE} not found in {CCO_CANDIDATES}",
        "Upload the .ttl to the volume's ontology/ folder via Catalog Explorer and re-run",
    )

# COMMAND ----------

# MAGIC %md ## 5. Predictive ML: scikit-learn, MLflow, Unity Catalog registry
# MAGIC Features are built in Spark SQL with a `CASE` of decimal literals on purpose: Spark returns `DECIMAL`, pandas then holds
# MAGIC `Decimal` objects, and MLflow's JSON logging fails on them unless they are cast to float. The training check covers that.

# COMMAND ----------

FEATURES = ["dist_m", "type_score", "confidence"]
RUN_ID, MODEL_NAME = None, f"{S}.feasibility_match_model"

with check(
    "scikit-learn training on Spark-built features (DECIMAL handled)",
    "%pip install scikit-learn; or use rule-based scoring",
):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    pdf = spark.sql(
        """SELECT id AS pair_id, CAST(rand(7) * 500 AS DOUBLE) AS dist_m,
                  CASE WHEN id % 3 = 0 THEN 0.5 WHEN id % 3 = 1 THEN 1.0 ELSE 0.8 END AS type_score,
                  CAST(rand(8) AS DOUBLE) AS confidence
           FROM range(2000)"""
    ).toPandas()
    print("dtypes before cast:", dict(pdf.dtypes.astype(str)))
    pdf[FEATURES] = pdf[FEATURES].astype(
        "float64"
    )  # without this, log_model can fail on Decimal
    rng = np.random.default_rng(7)
    y = ((pdf.dist_m < 250) & (pdf.type_score >= 0.8)).astype(int)
    y = y.where(rng.random(len(pdf)) > 0.1, 1 - y)  # 10% label noise
    X = pdf[FEATURES]
    tr, te = next(
        GroupShuffleSplit(test_size=0.25, random_state=7).split(
            X, y, groups=pdf.pair_id % 200
        )
    )
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=7).fit(
        X.iloc[tr], y.iloc[tr]
    )
    p = clf.predict_proba(X.iloc[te])[:, 1]
    metrics = {
        "auc": roc_auc_score(y.iloc[te], p),
        "avg_precision": average_precision_score(y.iloc[te], p),
    }
    print(metrics)
    assert metrics["auc"] > 0.7, f"model barely better than chance: {metrics}"

if "clf" in globals():
    with check(
        "MLflow log model with explicit signature",
        "Skip logging; keep the in-memory model",
    ):
        import mlflow
        from mlflow.models import infer_signature

        sig = infer_signature(X.iloc[:5], clf.predict(X.iloc[:5]))
        with mlflow.start_run(run_name="feasibility_match_model") as run:
            mlflow.log_params(
                {
                    "model": "HistGradientBoostingClassifier",
                    "features": ",".join(FEATURES),
                }
            )
            mlflow.log_metrics(metrics)
            # MLflow 3 saves with skops, which rejects classes it doesn't know unless trusted.
            mlflow.sklearn.log_model(
                clf,
                name="model",
                signature=sig,
                input_example=X.iloc[:5],
                skops_trusted_types=[
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
                ],
            )
        RUN_ID = run.info.run_id
        print("RUN_ID:", RUN_ID)
else:
    skip(
        "MLflow log model with explicit signature",
        "training check did not produce a model",
        "Fix training first",
    )

if RUN_ID:
    with check(
        "Register model in Unity Catalog",
        "Load the model from the MLflow run (runs:/<id>/model) instead",
    ):
        mlflow.set_registry_uri("databricks-uc")
        mv = mlflow.register_model(f"runs:/{RUN_ID}/model", MODEL_NAME)
        NOTES["uc_model_version"] = f"{MODEL_NAME} v{mv.version}"
        print("registered version:", mv.version)
else:
    skip(
        "Register model in Unity Catalog",
        "no MLflow run from the logging check",
        "Fix logging first",
    )

# COMMAND ----------

# MAGIC %md ## 6. Iterative clustering (label propagation) in Spark SQL
# MAGIC Connected components via iterative min-label propagation, no GraphFrames. The toy graph has components
# MAGIC {a, b, c}, {d, e} and the isolated node {f}.

# COMMAND ----------

MAX_ITER = 14

with check(
    "Iterative label propagation converges to the right components",
    "Cluster in pandas/networkx on the small set",
):
    nodes = ["a", "b", "c", "d", "e", "f"]
    edges = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("d", "e"), ("e", "d")]
    spark.createDataFrame([(n,) for n in nodes], "id STRING").write.mode(
        "overwrite"
    ).saveAsTable(f"{S}.t_cc_nodes")
    spark.createDataFrame(edges, "src STRING, dst STRING").write.mode(
        "overwrite"
    ).saveAsTable(f"{S}.t_cc_edges")
    spark.sql(
        f"CREATE OR REPLACE TABLE {S}.t_cc_0 AS SELECT id, id AS label FROM {S}.t_cc_nodes"
    )
    converged, cur = False, f"{S}.t_cc_0"
    for i in range(1, MAX_ITER + 1):
        prev, cur = f"{S}.t_cc_{(i - 1) % 2}", f"{S}.t_cc_{i % 2}"
        spark.sql(
            f"""CREATE OR REPLACE TABLE {cur} AS
                SELECT id, min(label) AS label FROM (
                  SELECT id, label FROM {prev}
                  UNION ALL
                  SELECT e.dst AS id, l.label FROM {prev} l JOIN {S}.t_cc_edges e ON l.id = e.src) t
                GROUP BY id"""
        )
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
        f"did not converge in {MAX_ITER} iterations; raise MAX_ITER or use networkx"
    )
    labels = {r.id: r.label for r in spark.table(cur).collect()}
    assert labels["a"] == labels["b"] == labels["c"] and labels["d"] == labels["e"]
    assert len({labels["a"], labels["d"], labels["f"]}) == 3, (
        f"components merged incorrectly: {labels}"
    )

# COMMAND ----------

# MAGIC %md ## 7. Write-back (`MERGE`)

# COMMAND ----------

with check(
    "MERGE update + insert on a Delta table",
    "INSERT OVERWRITE a staging 'outbox' table instead",
):
    spark.sql(
        f"CREATE OR REPLACE TABLE {S}.t_merge_target AS SELECT * FROM VALUES ('a', 1), ('b', 2) AS t(id, n)"
    )
    spark.sql(
        f"CREATE OR REPLACE TABLE {S}.t_merge_source AS SELECT * FROM VALUES ('b', 10), ('c', 3) AS t(id, n)"
    )
    spark.sql(
        f"""MERGE INTO {S}.t_merge_target t USING {S}.t_merge_source s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.n = t.n + s.n
            WHEN NOT MATCHED THEN INSERT (id, n) VALUES (s.id, s.n)"""
    )
    got = {r.id: r.n for r in spark.table(f"{S}.t_merge_target").collect()}
    assert got == {"a": 1, "b": 12, "c": 3}, f"unexpected MERGE result: {got}"

# COMMAND ----------

# MAGIC %md ## 8. Governance: row filter

# COMMAND ----------

with check(
    "Unity Catalog row filter", "Secure view filtering on is_account_group_member()"
):
    spark.sql(
        f"""CREATE OR REPLACE TABLE {S}.t_row_filter AS
            SELECT * FROM VALUES (1, 'OPEN'), (2, 'RESTRICTED'), (3, 'OPEN') AS t(id, marking)"""
    )
    spark.sql(
        f"CREATE OR REPLACE FUNCTION {S}.t_marking_filter(marking STRING) RETURN marking = 'OPEN' OR is_account_group_member('admins')"
    )
    try:
        spark.sql(
            f"ALTER TABLE {S}.t_row_filter SET ROW FILTER {S}.t_marking_filter ON (marking)"
        )
        print("visible rows with filter:", spark.table(f"{S}.t_row_filter").count())
    finally:
        spark.sql(f"ALTER TABLE {S}.t_row_filter DROP ROW FILTER")
    print(
        "A single-user workspace cannot show two identities seeing different rows; this only proves the DDL works."
    )

# COMMAND ----------

# MAGIC %md ## 9. Map rendering (`folium`)

# COMMAND ----------

with check(
    "folium map via displayHTML",
    "Native notebook map visualization, or a scatter plot of lat/lon",
):
    import folium

    m = folium.Map(location=[39.25, -105.25], zoom_start=15)
    folium.Marker(
        [39.25, -105.25], tooltip="center", icon=folium.Icon(color="red")
    ).add_to(m)
    for i, (la, lo) in enumerate([(39.2502, -105.2503), (39.2497, -105.2498)]):
        folium.CircleMarker([la, lo], radius=6, tooltip=f"point {i}").add_to(m)
    displayHTML(m._repr_html_())

# COMMAND ----------

# MAGIC %md ## 10. LLM access via `ai_query`
# MAGIC The endpoint list is a guess; the working one is recorded in `NOTES["ai_query_endpoint"]`.

# COMMAND ----------

ENDPOINTS = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-claude-sonnet-4",
    "databricks-gpt-oss-20b",
]
ENDPOINT_LOG = []  # (endpoint, status) for every endpoint tried
WORKING_EP = None

with check(
    "ai_query on a Foundation Model endpoint",
    "Genie space only; or template-based summaries",
):
    last_err = None
    for ep in ENDPOINTS:
        try:
            r = (
                spark.sql(
                    f"SELECT ai_query('{ep}', 'Reply with the single word OK') AS r"
                )
                .first()
                .r
            )
            print(f"{ep}: {r}")
            ENDPOINT_LOG.append((ep, "OK"))
            WORKING_EP = ep
            NOTES["ai_query_endpoint"] = ep
            break
        except Exception as e:
            last_err = e
            ENDPOINT_LOG.append(
                (ep, f"unavailable: {type(e).__name__}: {str(e)[:120]}")
            )
            print(f"{ep}: unavailable")
    else:
        raise last_err

if WORKING_EP:
    with check(
        "ai_query summary restricted to supplied rows, with citations",
        "Template-based summary (no LLM)",
    ):
        rows = [
            ("R1", "Truck seen near the north gate at 08:10."),
            ("R2", "Two vehicles seen near the north gate at 08:40."),
        ]
        context = "\n".join(f"[{i}] {t}" for i, t in rows)
        prompt = (
            "Using ONLY the reports below, write a 2-sentence summary. Cite report IDs in brackets. "
            f"Synthetic data.\n\nReports:\n{context}"
        )
        out = (
            spark.sql(
                f"SELECT ai_query('{WORKING_EP}', :prompt) AS s",
                args={"prompt": prompt},
            )
            .first()
            .s
        )
        print(out)
        cited = [i for i, _ in rows if f"[{i}]" in out]
        NOTES["ai_query_cited_ids"] = cited
        print("cited IDs found in output:", cited)
else:
    skip(
        "ai_query summary restricted to supplied rows, with citations",
        "no working endpoint",
        "Template-based summary (no LLM)",
    )

# COMMAND ----------

# MAGIC %md ## 11. Genie space API (scripted create and update)
# MAGIC Calls the REST endpoints through the SDK's API client, so it works whatever `databricks-sdk` version serverless ships.
# MAGIC Creates a throwaway space on its own table, exports it, updates it with the etag, asks one question, then trashes it.

# COMMAND ----------

import uuid  # noqa: E402

from databricks.sdk import WorkspaceClient  # noqa: E402

GENIE = "/api/2.0/genie/spaces"
GENIE_TITLE = "feasibility_genie_check"
w = WorkspaceClient()
genie_space_id = None
genie_warehouse = None

with check(
    "SQL warehouse available for Genie",
    "Create a SQL warehouse in the UI (Free Edition ships a starter warehouse)",
):
    whs = list(w.warehouses.list())
    assert whs, "no SQL warehouse in this workspace"
    genie_warehouse = whs[0].id
    NOTES["genie_warehouse"] = f"{whs[0].name} ({genie_warehouse})"
    print(NOTES["genie_warehouse"])

if genie_warehouse:
    with check(
        "Genie space create, export and update via REST API",
        "Create the space by hand from demo/genie-examples.md",
    ):
        spark.sql(f"""
          CREATE OR REPLACE TABLE {S}.t_genie AS
          SELECT * FROM VALUES ('Truck', 3), ('Aircraft', 2) AS t(object_type, n)""")
        payload = {
            "version": 2,
            "config": {
                "sample_questions": [
                    {"id": uuid.uuid4().hex, "question": ["How many objects in total?"]}
                ]
            },
            "data_sources": {"tables": [{"identifier": f"{S}.t_genie"}]},
            "instructions": {
                "text_instructions": [
                    {"id": uuid.uuid4().hex, "content": ["Synthetic data."]}
                ],
                "example_question_sqls": [
                    {
                        "id": uuid.uuid4().hex,
                        "question": ["How many objects by type?"],
                        "sql": [f"SELECT object_type, n FROM {S}.t_genie"],
                    }
                ],
            },
        }
        for sp in w.api_client.do("GET", GENIE).get("spaces", []):  # leftovers
            if sp.get("title") == GENIE_TITLE:
                w.api_client.do("DELETE", f"{GENIE}/{sp['space_id']}")
        created = w.api_client.do(
            "POST",
            GENIE,
            body={
                "warehouse_id": genie_warehouse,
                "title": GENIE_TITLE,
                "serialized_space": json.dumps(payload),
            },
        )
        genie_space_id = created["space_id"]
        got = w.api_client.do(
            "GET",
            f"{GENIE}/{genie_space_id}",
            query={"include_serialized_space": "true"},
        )
        exported = json.loads(got["serialized_space"])
        NOTES["genie_serialized_keys"] = sorted(exported)
        NOTES["genie_table_entry"] = exported["data_sources"]["tables"][0]
        print("exported keys:", sorted(exported))
        assert exported["instructions"]["example_question_sqls"], "example SQL lost"
        exported["instructions"]["text_instructions"][0]["content"] = ["Updated."]
        w.api_client.do(
            "PATCH",
            f"{GENIE}/{genie_space_id}",
            body={"serialized_space": json.dumps(exported), "etag": got.get("etag")},
        )
        again = json.loads(
            w.api_client.do(
                "GET",
                f"{GENIE}/{genie_space_id}",
                query={"include_serialized_space": "true"},
            )["serialized_space"]
        )
        assert again["instructions"]["text_instructions"][0]["content"] == [
            "Updated."
        ], "update did not apply"
else:
    skip(
        "Genie space create, export and update via REST API",
        "no SQL warehouse",
        "Create the space by hand from demo/genie-examples.md",
    )

if genie_space_id:
    with check(
        "Genie answers a question in the API-created space",
        "Ask the question in the Genie UI; or show the SQL",
    ):
        ans = w.genie.start_conversation_and_wait(
            genie_space_id, "How many objects are there in total?"
        )
        print(ans.as_dict().get("attachments"))
        assert ans.status.value == "COMPLETED", f"Genie status {ans.status}"
    w.api_client.do("DELETE", f"{GENIE}/{genie_space_id}")  # trash the throwaway space
else:
    skip(
        "Genie answers a question in the API-created space",
        "space was not created",
        "Ask the question in the Genie UI; or show the SQL",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Manual checks (UI)
# MAGIC - [ ] **Genie:** create a Genie space on any table in `feasibility_checks` (for example `t_merge_target`); ask a count question in plain English.
# MAGIC - [ ] **Databricks Apps:** create an app from the Streamlit or Dash template; confirm it deploys and can query a table through the SQL warehouse.
# MAGIC - [ ] **App map:** confirm folium or pydeck renders inside the app.
# MAGIC - [ ] **Lineage:** open `t_cc_1` (or `t_merge_target`) in Catalog Explorer and confirm the lineage graph renders.
# MAGIC - [ ] **AI/BI dashboard:** create a dashboard on a table in the schema; confirm the visualization you need (for example a point map) works and that a Genie space can be attached.
# MAGIC - [ ] **MLflow:** open the notebook's experiment; confirm the `feasibility_match_model` run shows params, metrics and the model, and that `feasibility_match_model` appears under `feasibility_checks` in Catalog Explorer (UC registration).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Template: add a check for a new design decision
# MAGIC Copy the cell below, rename the check, and keep it self-contained (own data, own `t_*` tables).

# COMMAND ----------

# with check("<what the design needs>", "<fallback if it fails>"):
#     spark.sql(f"CREATE OR REPLACE TABLE {S}.t_<name> AS SELECT * FROM VALUES (1), (2) AS t(x)")
#     assert spark.table(f"{S}.t_<name>").count() == 2

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

display(
    spark.createDataFrame(
        RESULTS,
        "check STRING, status STRING, seconds DOUBLE, error STRING, fallback STRING",
    )
)
print("NOTES:", json.dumps(NOTES, indent=2, default=str))
print("ai_query endpoints tried:", ENDPOINT_LOG)

# COMMAND ----------

# MAGIC %md ## Cleanup (uncomment to remove everything)

# COMMAND ----------

# spark.sql(f"DROP SCHEMA IF EXISTS {S} CASCADE")
