"""Schema reset and incremental, idempotent ingest of the landing files with ``COPY INTO``."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from .config import Config

# Rebuilt from the landing files on every run; every other table is CREATE OR REPLACE.
RESET_TABLES = [
    "bronze_observations",
    "bronze_tasking",
    "bronze_reports",
    "review_decisions",
]


def reset(spark: SparkSession, cfg: Config) -> dict:
    """Create the schema and volume, empty the landing folders and drop the bronze tables.

    Also drops ``review_decisions``, so analyst decisions from an earlier run are lost.

    :return: no facts (empty dict).
    """
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.s}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.s}.landing")
    os.makedirs(f"{cfg.vol}/ontology", exist_ok=True)
    # An orphaned stream from an earlier run would keep reading the landing files.
    for stream in spark.streams.active:
        stream.stop()
    for path in cfg.landing.values():
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    for tbl in RESET_TABLES:
        spark.sql(f"DROP TABLE IF EXISTS {cfg.s}.{tbl}")
    return {}


def copy_into_observations(spark: SparkSession, cfg: Config) -> None:
    """``COPY INTO bronze_observations`` from the landing folder; files already loaded are skipped."""
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {cfg.s}.bronze_observations (
        obs_id STRING, obs_time TIMESTAMP, lat DOUBLE, lon DOUBLE, reported_type STRING,
        confidence DOUBLE, producer STRING, sensor STRING, true_object_id STRING,
        source_file STRING, ingested_at TIMESTAMP)""")
    r = spark.sql(f"""
      COPY INTO {cfg.s}.bronze_observations
      FROM (SELECT obs_id, CAST(obs_time AS TIMESTAMP) AS obs_time,
                   CAST(lat AS DOUBLE) AS lat, CAST(lon AS DOUBLE) AS lon, reported_type,
                   CAST(confidence AS DOUBLE) AS confidence, producer, sensor, true_object_id,
                   _metadata.file_name AS source_file, current_timestamp() AS ingested_at
            FROM '{cfg.landing["observations"]}')
      FILEFORMAT = JSON""").first()
    print("COPY INTO bronze_observations:", r.asDict() if r else None)


def _landed_lines(folder: str) -> int:
    """Number of JSON lines across the files in a landing folder."""
    return sum(len(p.read_text().splitlines()) for p in Path(folder).glob("*.json"))


def ingest(spark: SparkSession, cfg: Config) -> dict:
    """Load observations in two deliveries, then reports and tasking.

    Batch 1 is loaded, batch 2 is moved from ``staged`` into the landing folder and the same
    ``COPY INTO`` loads only the new file; a third run loads nothing. Fails if any load
    duplicates rows.

    :return: no facts (empty dict).
    """
    obs_t = f"{cfg.s}.bronze_observations"
    copy_into_observations(spark, cfg)
    print("rows after batch 1:", spark.table(obs_t).count())

    for p in Path(cfg.landing["staged"]).glob("*.json"):
        shutil.move(p, Path(cfg.landing["observations"]) / p.name)
    copy_into_observations(spark, cfg)
    expected, n = _landed_lines(cfg.landing["observations"]), spark.table(obs_t).count()
    print("rows after batch 2 (batch 1 skipped):", n)
    assert n == expected, (
        f"expected {expected}, got {n} (duplicates mean files were reloaded)"
    )
    copy_into_observations(spark, cfg)
    assert spark.table(obs_t).count() == expected, (
        "COPY INTO re-run changed the row count"
    )

    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {cfg.s}.bronze_reports (
        report_id STRING, obs_id STRING, report_time TIMESTAMP, report_text STRING,
        source_file STRING, ingested_at TIMESTAMP)""")
    spark.sql(f"""
      COPY INTO {cfg.s}.bronze_reports
      FROM (SELECT report_id, obs_id, CAST(report_time AS TIMESTAMP) AS report_time, report_text,
                   _metadata.file_name AS source_file, current_timestamp() AS ingested_at
            FROM '{cfg.landing["reports"]}')
      FILEFORMAT = JSON""")
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {cfg.s}.bronze_tasking (
        task_id STRING, sensor STRING, min_lat DOUBLE, max_lat DOUBLE, min_lon DOUBLE, max_lon DOUBLE,
        start_time TIMESTAMP, end_time TIMESTAMP, requested_by STRING, priority INT,
        source_file STRING, ingested_at TIMESTAMP)""")
    spark.sql(f"""
      COPY INTO {cfg.s}.bronze_tasking
      FROM (SELECT task_id, sensor, CAST(min_lat AS DOUBLE) AS min_lat, CAST(max_lat AS DOUBLE) AS max_lat,
                   CAST(min_lon AS DOUBLE) AS min_lon, CAST(max_lon AS DOUBLE) AS max_lon,
                   CAST(start_time AS TIMESTAMP) AS start_time, CAST(end_time AS TIMESTAMP) AS end_time,
                   requested_by, CAST(priority AS INT) AS priority,
                   _metadata.file_name AS source_file, current_timestamp() AS ingested_at
            FROM '{cfg.landing["tasking"]}')
      FILEFORMAT = CSV FORMAT_OPTIONS ('header' = 'true')""")
    for t in ["bronze_observations", "bronze_reports", "bronze_tasking"]:
        print(t, spark.table(f"{cfg.s}.{t}").count(), "rows")
    return {}


def check_h3(spark: SparkSession, cfg: Config) -> None:
    """Fail fast if the H3 functions blocking relies on are missing or behave unexpectedly."""
    ring = (
        spark.sql(
            f"SELECT size(h3_kring(h3_longlatash3(-105.25, 39.25, {cfg.h3_res}), 1)) AS n"
        )
        .first()
        .n
    )
    assert ring == 7, f"unexpected H3 ring size {ring}"
