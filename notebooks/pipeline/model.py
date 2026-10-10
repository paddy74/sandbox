"""Match model (Layer 2): candidate pairs, training, registration and the three-way decision."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .config import hav_sql
from .icd203 import rank_sql

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .config import Config

FEATURES = [
    "dist_m",
    "type_score",
    "type_missing",
    "likelihood_rank",
    "is_analyst",
    "n_candidates",
    "dist_gap_m",
    "days_since_last_seen",
]
MODEL_NAME = "obj_resolution_match_model"
MODEL_ALIAS = "champion"


def build_candidates(spark: SparkSession, cfg: Config) -> dict:
    """Write ``silver_candidate_pairs``: every plausible (observation, object) pair with features.

    Blocking keeps objects in the observation's H3 cell or its neighbours, within
    ``cfg.max_dist_m`` and with a compatible type. ``label`` comes from ``true_object_id`` and
    stands in for past analyst adjudications; ground truth is never a feature.
    ``likelihood_rank`` (1 to 7) is the source's ICD 203 likelihood term, so machine scores and
    analyst terms enter the model on one scale. Features are cast to DOUBLE because Spark
    ``CASE`` returns DECIMAL, which breaks MLflow's JSON input example.

    :return: no facts (empty dict).
    """
    # dist_gap_m compares with the nearest *other* candidate on distance; a gap on match
    # probability would need the model's own output as an input.
    spark.sql(f"""
    CREATE OR REPLACE TABLE {cfg.s}.silver_candidate_pairs AS
    WITH o AS (
      SELECT *, explode(h3_kring(h3_longlatash3(lon, lat, {cfg.h3_res}), 1)) AS cell
      FROM {cfg.s}.bronze_observations),
    b AS (SELECT *, h3_longlatash3(lon, lat, {cfg.h3_res}) AS cell FROM {cfg.s}.oms_objects WHERE source = 'OMS')
    SELECT *,
      CAST(count(*) OVER w AS DOUBLE) AS n_candidates,
      CAST(coalesce(dist_m - CASE WHEN dist_m = min(dist_m) OVER w
                                  THEN try_element_at(array_sort(collect_list(dist_m) OVER w), 2)
                                  ELSE min(dist_m) OVER w END, -{cfg.max_dist_m}) AS DOUBLE) AS dist_gap_m
    FROM (
      SELECT o.obs_id, b.object_id,
        {hav_sql("o.lat", "o.lon", "b.lat", "b.lon")} AS dist_m,
        CAST(CASE WHEN o.reported_type IS NULL THEN 0.5
                  WHEN o.reported_type = b.object_type THEN 1.0
                  WHEN ta.ancestor IS NOT NULL THEN 0.8 ELSE 0.0 END AS DOUBLE) AS type_score,
        CAST(CAST(o.reported_type IS NULL AS INT) AS DOUBLE) AS type_missing,
        CAST({rank_sql("o.likelihood")} AS DOUBLE) AS likelihood_rank,
        CAST(CAST(o.producer = 'analyst' AS INT) AS DOUBLE) AS is_analyst,
        CAST(greatest(0, (unix_timestamp(o.obs_time) - unix_timestamp(b.last_seen)) / 86400) AS DOUBLE)
          AS days_since_last_seen,
        CAST(coalesce(o.true_object_id = b.object_id, false) AS INT) AS label
      FROM o JOIN b ON o.cell = b.cell
      LEFT JOIN {cfg.s}.type_ancestors ta ON ta.type = b.object_type AND ta.ancestor = o.reported_type)
    WHERE type_score > 0 AND dist_m <= {cfg.max_dist_m}
    WINDOW w AS (PARTITION BY obs_id)""")
    return {}


def train(
    spark: SparkSession, cfg: Config
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Train on ``silver_candidate_pairs``, log to MLflow and register in Unity Catalog.

    The held-out split is grouped by observation, so no observation's candidates appear on
    both sides. The new version gets the ``champion`` alias that ``score`` loads.

    :return: the fitted classifier and cheat-card facts (held-out metrics, model version).
    """
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    pdf = spark.table(f"{cfg.s}.silver_candidate_pairs").toPandas()
    X, y = pdf[FEATURES].astype("float64"), pdf["label"]
    assert y.nunique() == 2, "candidate pairs need both match and non-match labels"
    train_idx, test_idx = next(
        GroupShuffleSplit(test_size=0.25, random_state=7).split(
            X, y, groups=pdf["obs_id"]
        )
    )
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=7).fit(
        X.iloc[train_idx], y.iloc[train_idx]
    )
    p_test = clf.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = {
        "held_out_auc": float(roc_auc_score(y.iloc[test_idx], p_test)),
        "held_out_avg_precision": float(
            average_precision_score(y.iloc[test_idx], p_test)
        ),
    }
    print("HELD-OUT (synthetic, grouped by observation):", metrics)

    signature = infer_signature(X.iloc[:5], clf.predict(X.iloc[:5]))
    with mlflow.start_run(run_name="obj_resolution_match_model") as run:
        mlflow.log_params(
            {"model": "HistGradientBoostingClassifier", "features": ",".join(FEATURES)}
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            clf, "model", signature=signature, input_example=X.iloc[:5]
        )
    mlflow.set_registry_uri("databricks-uc")
    name = f"{cfg.s}.{MODEL_NAME}"
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
    MlflowClient().set_registered_model_alias(name, MODEL_ALIAS, mv.version)
    facts = {k: round(v, 4) for k, v in metrics.items()}
    facts["uc_model_version"] = f"{name} v{mv.version}"
    print("registered:", facts["uc_model_version"])
    return clf, facts


def score(
    spark: SparkSession, cfg: Config, clf: HistGradientBoostingClassifier | None = None
) -> dict:
    """Score every candidate pair and write ``silver_scored_pairs`` and ``silver_model_decisions``.

    Each observation takes its best pair: ``AUTO`` at or above ``cfg.auto_t``, ``REVIEW`` down
    to ``cfg.review_t``, else (or with no candidate) ``NOMINATE``. Thresholds are demo values,
    not tuned. Band precision is in-sample against synthetic ground truth.

    :param clf: classifier from ``train``; ``None`` loads the registered ``champion`` version.
    :return: cheat-card facts: band counts and precision, seeded duplicates surfaced.
    """
    if clf is None:
        import mlflow

        mlflow.set_registry_uri("databricks-uc")
        clf = mlflow.sklearn.load_model(f"models:/{cfg.s}.{MODEL_NAME}@{MODEL_ALIAS}")

    pdf = spark.table(f"{cfg.s}.silver_candidate_pairs").toPandas()
    pdf[FEATURES] = pdf[FEATURES].astype("float64")
    pdf["match_prob"] = clf.predict_proba(pdf[FEATURES])[:, 1]
    spark.createDataFrame(
        pdf[["obs_id", "object_id", "match_prob"] + FEATURES]
    ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{cfg.s}.silver_scored_pairs"
    )
    spark.sql(f"""
    CREATE OR REPLACE TABLE {cfg.s}.silver_model_decisions AS
    WITH best AS (
      SELECT *, row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk
      FROM {cfg.s}.silver_scored_pairs)
    SELECT ob.obs_id, ob.true_object_id, b.object_id AS matched_object_id, round(b.match_prob, 3) AS match_prob,
           CASE WHEN b.match_prob >= {cfg.auto_t} THEN 'AUTO'
                WHEN b.match_prob >= {cfg.review_t} THEN 'REVIEW'
                ELSE 'NOMINATE' END AS decision
    FROM {cfg.s}.bronze_observations ob LEFT JOIN best b ON ob.obs_id = b.obs_id AND b.rk = 1""")

    # A NOMINATE is correct when the real object is not in the object system (or is noise:
    # NULL truth); a match is correct only when it names the real object.
    bands = spark.sql(f"""
      SELECT decision, count(*) AS n,
        round(avg(CASE WHEN decision = 'NOMINATE' THEN CAST(coalesce(true_object_id, '') NOT LIKE 'OBJ-%' AS DOUBLE)
                       ELSE CAST(coalesce(matched_object_id = true_object_id, false) AS DOUBLE) END), 3)
          AS precision_vs_synthetic_truth
      FROM {cfg.s}.silver_model_decisions GROUP BY decision ORDER BY decision""").toPandas()
    bands["pct"] = (100 * bands.n / bands.n.sum()).round(1)
    print("IN-SAMPLE band counts and precision (synthetic ground truth):")
    print(bands.to_string(index=False))
    n_review = int(bands.loc[bands.decision == "REVIEW", "n"].sum())
    print(
        f"REVIEW queue: {n_review} rows ({100 * n_review / bands.n.sum():.1f}% of observations); "
        "present it as a ranked queue, do not tune to shrink it"
    )

    # Does the model surface the seeded duplicate records as ambiguous (two close top scores)?
    flagged = spark.sql(f"""
      WITH r AS (SELECT obs_id, object_id, match_prob,
                        row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk
                 FROM {cfg.s}.silver_scored_pairs),
      t AS (SELECT least(a.object_id, b.object_id) o1, greatest(a.object_id, b.object_id) o2, a.obs_id
            FROM r a JOIN r b ON a.obs_id = b.obs_id AND a.rk = 1 AND b.rk = 2
            WHERE b.match_prob >= 0.3 AND a.match_prob - b.match_prob <= 0.2)
      SELECT o1, o2, count(*) n_obs FROM t GROUP BY o1, o2 HAVING count(*) >= 2""").toPandas()
    seeded = spark.sql(
        f"SELECT dup_of, object_id FROM {cfg.s}.oms_objects WHERE dup_of IS NOT NULL"
    ).toPandas()
    seeded_pairs = {
        tuple(sorted(p)) for p in zip(seeded.dup_of, seeded.object_id, strict=True)
    }
    flagged_pairs = {tuple(sorted(p)) for p in zip(flagged.o1, flagged.o2, strict=True)}
    recovered = f"{len(seeded_pairs & flagged_pairs)}/{len(seeded_pairs)}"
    print("seeded duplicate objects surfaced as ambiguous:", recovered)
    return {
        "bands": json.loads(bands.set_index("decision").to_json(orient="index")),
        "duplicates_recovered": recovered,
    }
