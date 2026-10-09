"""Cluster unmatched observations into new objects and write results back to the object system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import hav_sql
from .synthetic import DESIGNATOR_IRI

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from .config import Config

MAX_ITER = 14  # label propagation passes before the stage fails as not converged
# Two NOMINATE observations closer than this, with compatible types, belong to one object.
LINK_M = 400


def nominate(spark: SparkSession, cfg: Config) -> dict:
    """Group ``NOMINATE`` observations into ``gold_nominations``, one row per new object.

    Connected components by iterative min-label propagation in Spark SQL (no GraphFrames):
    each observation takes the smallest label among its neighbours until nothing changes.
    Clusters of two or more observations become nominations; singletons stay unexplained.

    :return: cheat-card facts: nominations and observations recovered.
    """
    s = cfg.s
    spark.sql(f"""
    CREATE OR REPLACE TABLE {s}.nom_obs AS
    SELECT b.obs_id, b.lat, b.lon, b.reported_type
    FROM {s}.silver_model_decisions d JOIN {s}.bronze_observations b ON d.obs_id = b.obs_id
    WHERE d.decision = 'NOMINATE'""")
    spark.sql(f"""
    CREATE OR REPLACE TABLE {s}.nom_edges AS
    WITH a AS (SELECT *, explode(h3_kring(h3_longlatash3(lon, lat, {cfg.h3_res}), 1)) AS cell FROM {s}.nom_obs),
    c AS (SELECT *, h3_longlatash3(lon, lat, {cfg.h3_res}) AS cell FROM {s}.nom_obs)
    SELECT a.obs_id AS src, c.obs_id AS dst
    FROM a JOIN c ON a.cell = c.cell AND a.obs_id <> c.obs_id
    WHERE {hav_sql("a.lat", "a.lon", "c.lat", "c.lon")} <= {LINK_M}
      AND (a.reported_type IS NULL OR c.reported_type IS NULL OR a.reported_type = c.reported_type)""")
    spark.sql(
        f"CREATE OR REPLACE TABLE {s}.cc_0 AS SELECT obs_id AS id, obs_id AS label FROM {s}.nom_obs"
    )

    converged, cur = False, f"{s}.cc_0"
    # Alternate between cc_0 and cc_1 rather than creating a table per pass.
    for i in range(1, MAX_ITER + 1):
        prev, cur = f"{s}.cc_{(i - 1) % 2}", f"{s}.cc_{i % 2}"
        spark.sql(f"""
          CREATE OR REPLACE TABLE {cur} AS
          SELECT id, min(label) AS label FROM (
            SELECT id, label FROM {prev}
            UNION ALL
            SELECT e.dst AS id, l.label FROM {prev} l JOIN {s}.nom_edges e ON l.id = e.src) t
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

    spark.sql(f"CREATE OR REPLACE TABLE {s}.cc_final AS SELECT * FROM {cur}")
    spark.sql(f"""
    CREATE OR REPLACE TABLE {s}.silver_nom_members AS
    SELECT l.id AS obs_id, concat('NOM-', substr(sha2(l.label, 256), 1, 10)) AS object_id
    FROM {s}.cc_final l
    WHERE l.label IN (SELECT label FROM {s}.cc_final GROUP BY label HAVING count(*) >= 2)""")
    spark.sql(f"""
    CREATE OR REPLACE TABLE {s}.gold_nominations AS
    SELECT *, concat('New-', lpad(CAST(row_number() OVER (ORDER BY object_id) AS STRING), 3, '0')) AS designator,
           '{DESIGNATOR_IRI}' AS designator_type
    FROM (SELECT m.object_id,
                 coalesce(max(c.reported_type), 'Unknown') AS object_type,
                 avg(c.lat) AS lat, avg(c.lon) AS lon, count(*) AS obs_count
          FROM {s}.silver_nom_members m JOIN {s}.nom_obs c ON m.obs_id = c.obs_id
          GROUP BY m.object_id)""")
    r = spark.sql(f"""
      SELECT (SELECT count(*) FROM {s}.gold_nominations) noms,
             (SELECT coalesce(sum(obs_count), 0) FROM {s}.gold_nominations) recovered,
             (SELECT count(*) FROM {s}.bronze_observations) n_obs""").first()
    print(
        f"nominations={r.noms}, observations recovered (would otherwise be discarded)={r.recovered}"
    )
    return {
        "nominations": r.noms,
        "observations_recovered": int(r.recovered),
        "recovered_share_of_observations_pct": round(100 * r.recovered / r.n_obs, 1),
    }


def write_back(spark: SparkSession, cfg: Config) -> dict:
    """``MERGE`` model results into ``oms_objects``.

    ``AUTO`` associations add to each object's ``obs_count`` and refresh ``last_seen``;
    nominations are inserted as ``NOMINATED`` objects. Not idempotent on its own: running it
    twice double-counts ``obs_count``, so re-run from ``synthetic.write_sources``.

    :return: cheat-card facts: object counts before and after.
    """
    before = spark.table(f"{cfg.s}.oms_objects").count()
    spark.sql(f"""
    MERGE INTO {cfg.s}.oms_objects t
    USING (SELECT matched_object_id AS object_id, count(*) AS n
           FROM {cfg.s}.silver_model_decisions WHERE decision = 'AUTO' GROUP BY matched_object_id) s
    ON t.object_id = s.object_id
    WHEN MATCHED THEN UPDATE SET t.obs_count = t.obs_count + s.n, t.last_seen = current_timestamp()""")
    spark.sql(f"""
    MERGE INTO {cfg.s}.oms_objects t
    USING {cfg.s}.gold_nominations s ON t.object_id = s.object_id
    WHEN NOT MATCHED THEN INSERT (object_id, object_type, lat, lon, marking, first_seen, last_seen,
                                  obs_count, source, status, designator, designator_type)
    VALUES (s.object_id, s.object_type, s.lat, s.lon, 'OPEN', current_timestamp(), current_timestamp(),
            s.obs_count, 'DATABRICKS_NOMINATION', 'NOMINATED', s.designator, s.designator_type)""")
    after = spark.table(f"{cfg.s}.oms_objects").count()
    print(f"OMS objects: {before} before, {after} after nominations")
    return {"objects_before_merge": before, "objects_after_merge": after}
