"""Analyst review: the ranked review queue and decision write-back."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .icd203 import CONFIDENCE, likelihood_sql

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from .config import Config


def create_review_views(spark: SparkSession, cfg: Config) -> dict:
    """Create ``review_decisions`` (if missing) and the ``review_queue`` and ``training_labels_from_review`` views.

    ``review_queue`` is the ``REVIEW`` band with the top candidate, the runner-up and the
    features behind the score; ``*_likelihood`` columns give the ICD 203 term for each numeric
    score, which is what the app shows. Each analyst decision, with the analyst's confidence,
    becomes a training label for the next retrain; the retrain is described in the demo, not run.

    :return: no facts (empty dict).
    """
    s = cfg.s
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s}.review_decisions (
        obs_id STRING, object_id STRING, decision STRING, confidence STRING, decided_by STRING,
        decided_at TIMESTAMP)""")
    spark.sql(f"""
    CREATE OR REPLACE VIEW {s}.review_queue AS
    WITH ranked AS (
      SELECT *, row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk FROM {s}.silver_scored_pairs)
    SELECT d.obs_id, d.matched_object_id AS candidate_object_id, o.object_type AS candidate_type,
           d.match_prob, {likelihood_sql("d.match_prob")} AS match_likelihood, round(p1.dist_m, 0) AS dist_m, p1.type_score, CAST(p1.n_candidates AS INT) AS n_candidates,
           round(p1.days_since_last_seen, 1) AS days_since_last_seen, b.reported_type, b.producer, b.sensor,
           b.confidence, b.likelihood, p2.object_id AS runner_up_object_id,
           round(p2.match_prob, 3) AS runner_up_prob, {likelihood_sql("p2.match_prob")} AS runner_up_likelihood,
           b.lat, b.lon, b.obs_time, rd.decision AS analyst_decision, rd.confidence AS analyst_confidence
    FROM {s}.silver_model_decisions d
    JOIN {s}.bronze_observations b ON b.obs_id = d.obs_id
    JOIN {s}.oms_objects o ON o.object_id = d.matched_object_id
    JOIN ranked p1 ON p1.obs_id = d.obs_id AND p1.rk = 1
    LEFT JOIN ranked p2 ON p2.obs_id = d.obs_id AND p2.rk = 2
    LEFT JOIN {s}.review_decisions rd ON rd.obs_id = d.obs_id
    WHERE d.decision = 'REVIEW'""")
    spark.sql(f"""
    CREATE OR REPLACE VIEW {s}.training_labels_from_review AS
    SELECT obs_id, object_id, CAST(decision = 'APPROVE' AS INT) AS label, confidence
    FROM {s}.review_decisions""")
    print("REVIEW queue rows:", spark.table(f"{s}.review_queue").count())
    return {}


def adjudicate(
    spark: SparkSession,
    cfg: Config,
    obs_id: str,
    object_id: str,
    decision: str,
    confidence: str,
    decided_by: str = "analyst_demo",
) -> None:
    """Record an analyst decision on one review item; the app's buttons do the same.

    A second decision on the same observation is ignored.

    :param decision: ``APPROVE`` (also adds the observation to the object and moves its
        ``last_seen`` up to the observation's ``obs_time``) or ``REJECT``.
    :param confidence: the analyst's ICD 203 confidence in the judgement: High, Moderate or Low.
    :raises ValueError: for any other decision or confidence.
    """
    if decision not in ("APPROVE", "REJECT"):
        raise ValueError(f"decision must be APPROVE or REJECT, got {decision!r}")
    if confidence not in CONFIDENCE:
        raise ValueError(f"confidence must be one of {CONFIDENCE}, got {confidence!r}")
    if spark.sql(
        f"SELECT 1 FROM {cfg.s}.review_decisions WHERE obs_id = '{obs_id}'"
    ).count():
        print(f"{obs_id} already adjudicated; no change")
        return
    spark.sql(f"""
      INSERT INTO {cfg.s}.review_decisions (obs_id, object_id, decision, confidence, decided_by, decided_at)
      VALUES ('{obs_id}', '{object_id}', '{decision}', '{confidence}', '{decided_by}', current_timestamp())""")
    if decision == "APPROVE":
        spark.sql(f"""
          MERGE INTO {cfg.s}.oms_objects t
          USING (SELECT '{object_id}' AS object_id, obs_time
                 FROM {cfg.s}.bronze_observations WHERE obs_id = '{obs_id}') s
          ON t.object_id = s.object_id
          WHEN MATCHED THEN UPDATE SET t.obs_count = t.obs_count + 1,
                                       t.last_seen = greatest(t.last_seen, s.obs_time)""")
    print(f"{decision} ({confidence} confidence): {obs_id} -> {object_id}")
