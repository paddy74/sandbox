# Databricks notebook source
# MAGIC %md
# MAGIC # Object resolution pipeline: observations to objects (synthetic)
# MAGIC
# MAGIC Runs every stage of the demo on **Databricks Free Edition (serverless)**. The code lives in `notebooks/pipeline/`; each
# MAGIC cell below calls one stage and shows its output.
# MAGIC
# MAGIC **What it builds, in layer order**
# MAGIC 1. **Governed data foundation:** four mocked source types (object system, tasking CSV, observation JSON, report text)
# MAGIC    land with `COPY INTO` plus `source_file` / `ingested_at`; the public CCO ontology is flattened to a Delta table
# MAGIC    and used for type compatibility.
# MAGIC 2. **Predictive ML:** H3 blocking, candidate pairs, a scikit-learn gradient boosting match model logged to MLflow and
# MAGIC    registered in Unity Catalog. Probability drives `AUTO` (>= 0.9), `REVIEW` (0.5 to 0.9) or `NOMINATE` (< 0.5 or no
# MAGIC    candidate); nominated observations are clustered by label propagation in Spark SQL.
# MAGIC 3. **GenAI:** a sourced object dossier with `ai_query`, with every citation checked against the retrieved reports.
# MAGIC 4. **Presentation:** a review queue with analyst write-back, views for the AI/BI dashboard, and the Genie space
# MAGIC    (created or updated from `demo/genie-examples.md`).
# MAGIC
# MAGIC **Running it.** Top to bottom is safe to repeat: section 1 resets the landing files and bronze tables, everything else is
# MAGIC `CREATE OR REPLACE`. To run **one stage**, run the setup cells, then that stage's cell; it reads its inputs from the
# MAGIC tables earlier stages wrote. Exception: section 8's `MERGE` adds to object counts, so re-run from section 2 instead.
# MAGIC
# MAGIC > All data is synthetic. `true_object_id` is ground truth for scoring the demo only; it is never a model feature and
# MAGIC > never appears in the dossier prompt. Model numbers printed here are **in-sample** unless labelled held-out.

# COMMAND ----------

# MAGIC %pip install -q rdflib folium

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import datetime as dt
import json
import time

from pipeline import (
    dossier,
    genie,
    heroes,
    ingest,
    model,
    nominate,
    ontology,
    present,
    review,
    synthetic,
)
from pipeline.config import Config

T0 = time.time()
cfg = Config.from_spark(spark)
notes: dict = {}  # cheat-card facts from each stage, printed in the last cell
print("Using", cfg.s, "| landing:", cfg.vol)

# COMMAND ----------

# MAGIC %md ## 1. Schema, volume and reset
# MAGIC Creates the schema and landing volume, empties the landing folders and drops the bronze tables and `review_decisions`.

# COMMAND ----------

notes |= ingest.reset(spark, cfg)

# COMMAND ----------

# MAGIC %md ## 2. Synthetic sources (Layer 1)
# MAGIC Writes `oms_objects` (the object system) and `seed_twins`, and drops observation, report and tasking files into the
# MAGIC landing volume. Seeded cases make the review queue real: twin records close together, duplicates of one real object,
# MAGIC objects missing from the object system, noise and cross-producer corroboration. Check the printed counts:
# MAGIC 2,025 objects.

# COMMAND ----------

src = synthetic.generate(dt.datetime.now(dt.UTC).replace(tzinfo=None, microsecond=0))
notes |= synthetic.write_sources(spark, cfg, src)

# COMMAND ----------

# MAGIC %md ## 3. Ingest with `COPY INTO` (incremental, idempotent)
# MAGIC Loads observation batch 1, delivers batch 2 and loads only that file, then shows a third run loads nothing.
# MAGIC No streaming query, checkpoint or long-running compute.

# COMMAND ----------

notes |= ingest.ingest(spark, cfg)

# COMMAND ----------

# MAGIC %md ## 4. H3 sanity check
# MAGIC Fails here, not mid-model, if the H3 functions used for blocking are unavailable.

# COMMAND ----------

ingest.check_h3(spark, cfg)

# COMMAND ----------

# MAGIC %md ## 5. CCO ontology: parse, flatten, resolve demo types to IRIs
# MAGIC Writes `cco_classes`, `type_iri` and `type_ancestors`. Type compatibility in the model comes from the ontology, not a
# MAGIC hand-built table.

# COMMAND ----------

notes |= ontology.load_cco(spark, cfg)
display(spark.sql(f"SELECT label, iri FROM {cfg.s}.type_iri ORDER BY label"))

# COMMAND ----------

# MAGIC %md ## 6. Candidate pairs and match model (Layer 2)
# MAGIC Pairs each observation with nearby objects of a compatible type (H3 resolution 8, ring 1, <= 500 m) and computes the
# MAGIC features in `model.FEATURES`. Labels simulate past analyst decisions. Check that both labels appear.

# COMMAND ----------

notes |= model.build_candidates(spark, cfg)
display(
    spark.sql(
        f"SELECT label, count(*) n FROM {cfg.s}.silver_candidate_pairs GROUP BY label"
    )
)

# COMMAND ----------

clf, facts = model.train(spark, cfg)
notes |= facts

# COMMAND ----------

# MAGIC %md ### Score every pair and apply the three-way decision
# MAGIC `AUTO` >= 0.9, `REVIEW` 0.5 to 0.9, `NOMINATE` < 0.5 or no candidate. Run alone, this loads the registered
# MAGIC `champion` model instead of the one trained above.

# COMMAND ----------

notes |= model.score(spark, cfg, globals().get("clf"))
display(
    spark.sql(
        f"SELECT decision, count(*) n FROM {cfg.s}.silver_model_decisions GROUP BY decision ORDER BY 1"
    )
)

# COMMAND ----------

# MAGIC %md ## 7. Cluster unmatched observations into nominations
# MAGIC Writes `gold_nominations`: clusters of two or more `NOMINATE` observations, each a candidate new object.

# COMMAND ----------

notes |= nominate.nominate(spark, cfg)

# COMMAND ----------

# MAGIC %md ## 8. Write-back to the object system (`MERGE`)
# MAGIC `AUTO` associations update their objects; nominations are inserted as new objects. Expect 2,025 before and
# MAGIC 2,025 + nominations after.

# COMMAND ----------

notes |= nominate.write_back(spark, cfg)
display(
    spark.sql(
        f"SELECT source, status, count(*) n FROM {cfg.s}.oms_objects GROUP BY ALL"
    )
)

# COMMAND ----------

# MAGIC %md ## 9. Hero objects for the demo
# MAGIC **A** clean auto-associate (Armored Fighting Vehicle, corroborated) · **B** twin-object review case ·
# MAGIC **C** nominated cluster of individually weak observations. Chosen from this run's data and stored in `demo_heroes`.

# COMMAND ----------

notes |= heroes.select_heroes(spark, cfg)
display(spark.table(f"{cfg.s}.demo_heroes"))

# COMMAND ----------

# MAGIC %md ## 10. Review queue and analyst write-back (Layer 4)
# MAGIC `review_queue` is the `REVIEW` band ranked by match probability, with the runner-up and the features behind the score.
# MAGIC Analyst decisions go to `review_decisions` and become training labels (`training_labels_from_review`) for the next
# MAGIC retrain, which the demo describes but does not run.

# COMMAND ----------

notes |= review.create_review_views(spark, cfg)
hero_b = heroes.load_heroes(spark, cfg)["B"]
display(
    spark.sql(f"SELECT * FROM {cfg.s}.review_queue ORDER BY match_prob DESC LIMIT 20")
)
print("Hero B before the decision:")
display(
    spark.sql(f"SELECT * FROM {cfg.s}.review_queue WHERE obs_id = '{hero_b.obs_id}'")
)

# COMMAND ----------

# Analyst approves the model's top candidate for hero B; its obs_count goes up by one.
count_sql = (
    f"SELECT obs_count FROM {cfg.s}.oms_objects WHERE object_id = '{hero_b.object_id}'"
)
before = spark.sql(count_sql).first().obs_count
review.adjudicate(spark, cfg, hero_b.obs_id, hero_b.object_id, "APPROVE")
print(
    f"{hero_b.object_id} obs_count: {before} -> {spark.sql(count_sql).first().obs_count}"
)
display(spark.table(f"{cfg.s}.review_decisions"))
display(spark.table(f"{cfg.s}.training_labels_from_review"))

# COMMAND ----------

# MAGIC %md ### 10b. Analyst decision form
# MAGIC Pick a pending review item, choose APPROVE or REJECT, set **confirm = yes**, then run the cell below the form.
# MAGIC The decision shows in the dashboard's review-queue table after a refresh. `confirm` defaults to `no`, so running the
# MAGIC whole notebook never submits a decision. Re-run the form cell to refresh the list of pending items.

# COMMAND ----------

pending = spark.sql(f"""
  SELECT obs_id, candidate_object_id, match_prob
  FROM {cfg.s}.review_queue WHERE analyst_decision IS NULL
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
    review.adjudicate(
        spark, cfg, *review_choices[choice], dbutils.widgets.get("decision")
    )
    dbutils.widgets.remove("confirm")  # back to "no", so the next run can't resubmit
    dbutils.widgets.dropdown(
        "confirm", "no", ["no", "yes"], "3. Confirm (yes to submit)"
    )
    print("Submitted. Re-run the form cell to refresh the pending list.")
display(spark.table(f"{cfg.s}.review_decisions"))

# COMMAND ----------

# MAGIC %md ## 11. Sourced object dossier (Layer 3)
# MAGIC Writes `object_dossiers` for heroes A and B. Each dossier cites only reports retrieved for that object; an `ai_query`
# MAGIC error or a bad citation falls back to a templated dossier, labelled as such in `generated_by`.

# COMMAND ----------

notes |= dossier.build_hero_dossiers(spark, cfg)

# COMMAND ----------

# MAGIC %md ## 12. Governance: row filter on synthetic markings
# MAGIC Single-user workspace, so two identities cannot be shown. This demonstrates the mechanism; the filter is dropped again
# MAGIC so dashboards and Genie see all rows.

# COMMAND ----------

notes |= present.demo_row_filter(spark, cfg)

# COMMAND ----------

# MAGIC %md ## 13. Map for hero A (folium)
# MAGIC The object (red) and the observations auto-associated with it.

# COMMAND ----------

displayHTML(
    present.hero_map(
        spark, cfg, heroes.load_heroes(spark, cfg)["A"].object_id
    )._repr_html_()
)

# COMMAND ----------

# MAGIC %md ## 14. Views for the AI/BI dashboard and Genie
# MAGIC Point the dashboard at `dash_map_points`, `dash_decisions`, `dash_object_summary` and `review_queue`.

# COMMAND ----------

notes |= present.create_dashboard_views(spark, cfg)
display(spark.table(f"{cfg.s}.dash_decisions"))
display(spark.table(f"{cfg.s}.dash_object_summary"))

# COMMAND ----------

# MAGIC %md ## 15. Genie space (create or update)
# MAGIC Builds the space from `demo/genie-examples.md` and creates it, or updates the space with the same title, keeping
# MAGIC settings made in the UI such as hidden columns. If this cell fails, create the space by hand from the same file.

# COMMAND ----------

notes |= genie.sync_space(spark, cfg)

# COMMAND ----------

# MAGIC %md ## Cheat card
# MAGIC Numbers to read off during the demo. All synthetic; band precision is in-sample, AUC and average precision are held-out.

# COMMAND ----------

notes["elapsed_minutes"] = round((time.time() - T0) / 60, 1)
print(json.dumps(notes, indent=2, default=str))
display(spark.table(f"{cfg.s}.demo_heroes"))
