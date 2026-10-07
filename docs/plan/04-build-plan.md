# 04: Build plan

Purpose: the minimum viable build, a 60-minute build sequence with checkpoints, ordered cut lines, a stall rule, and the checks to complete before a demo. Table names, thresholds and hero objects come from files 02 and 03.

Build order is fixed: **data → model → GenAI → presentation**. Never cut a layer. A thin layer beats a missing layer, because the demo is one cohesive solution across all four.

**Status:** `notebooks/generate_data.py` ran end to end in 4.8 minutes of compute. Held-out AUC 0.859; AUTO / REVIEW / NOMINATE 66.6% / 18.1% / 15.3%; 146 nominations; heroes in `demo_heroes`. Full numbers are in file 02 and file 03, section 6.

**REVIEW queue decision:** the REVIEW queue is large (810 rows, 18.1%) because of weakly typed observations, not the seeded twins. **Decision: do not tune seeds or thresholds to shrink it.** Present the queue as ranked by match probability (measured numbers are in file 03, section 6).

## 1. Build

Every cell is idempotent (`CREATE OR REPLACE`, `COPY INTO`, a reset cell), and the notebook has a markdown cell at the top saying what it does. Build-minute `B:00` is the start of the build.

### Minimum viable build (MVB)

This is the list that must exist at the end of the build. Anything not on it is optional.

| #   | MVB item                                                                                                                                                             | Layer  | Notes                                                                                             |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| 1   | Synthetic OMS objects (Delta), observation JSON, tasking CSV and report JSON in a UC volume, with the seeded cases from file 03 (twins, partial observations, noise) | 1      | Seeds are part of the MVB, not polish. Without them the REVIEW queue is empty and the story fails |
| 2   | **`COPY INTO`** to `bronze_observations` (and tasking, reports), with `source_file` and `ingested_at`                                                                | 1      | Not Auto Loader                                                                                   |
| 3   | CCO flatten via rdflib to `cco_classes`, type compatibility from `ancestor_iris`                                                                                     | 1      | Fallback: hand-built `type_ancestors`                                                             |
| 4   | **H3 blocking** (res 8, `h3_kring` 1, ≤ 500 m) and `silver_candidate_pairs`                                                                                          | 2      | Features cast to DOUBLE at creation                                                               |
| 5   | Model **trained and scored in memory first** with every feature cast to **DOUBLE**; print AUC, average precision                                                     | 2      | Logging and registration come after, never before                                                 |
| 6   | **Three-way decision** in `silver_model_decisions` (`AUTO` ≥ 0.9, `REVIEW` 0.5 to 0.9, `NOMINATE` < 0.5 or none), band counts, precision per band (synthetic)        | 2      | Print the counts: they are the demo numbers                                                       |
| 7   | **Label propagation** over `NOMINATE` observations to `gold_nominations` (≥ 2 observations), with a convergence check that fails loudly                              | 2      | Spark SQL, not GraphFrames                                                                        |
| 8   | **MERGE** write-back to `oms_objects`, plus one approval cell for hero B writing `review_decisions`                                                                  | 2 to 4 | Closes the loop                                                                                   |
| 9   | **Sourced `ai_query` dossier** for heroes A and B, bracketed IDs, a check that cited IDs exist in the retrieved set                                                  | 3      | Template dossier is the fallback                                                                  |
| 10  | **Databricks App** deployed from `app/` (SQL warehouse resource `sql-warehouse`, schema grants per the `app/app.py` docstring), plus the notebook walkthrough order  | 4      | Fallback: AI/BI dashboard (point map, review queue table, counts per decision)                    |

Optional, in cut order: Genie space; MLflow logging and UC registration (logging is wanted, registration is the first model item to cut); dashboard polish. If the App will not deploy, switch to the AI/BI dashboard fallback rather than skipping the layer.

### Minute-by-minute (60 minutes, zero slack by design)

The slack is the cut lines: the Genie space (4 minutes, moved to presentation prep) and UC registration (3 minutes) are the reclaimable time.

| Build min    | Min | Step                         | Concrete deliverable                                                                                                                                                                                                                                    | Checkpoint                                                                                                                                    |
| ------------ | --- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| B:00 to B:05 | 5   | Setup                        | Markdown header cell; `%pip install` for rdflib and folium (restarts Python, so do this first); schema, volume, reset cell; config cell with `S` and thresholds                                                                                         | Schema and volume exist                                                                                                                       |
| B:05 to B:20 | 15  | Data foundation              | Generate OMS objects, observations (including twins, partial observations, noise, markings), tasking, reports; write files to the volume; `COPY INTO` three bronze tables; print row counts per source                                                  | **B:20: every bronze table has rows; seeded cases present; counts printed.** If behind, drop tasking and reports to the end                   |
| B:20 to B:25 | 5   | CCO and type compatibility   | rdflib parse to `cco_classes` with `ancestor_iris`; confirm the five leaf types and three generics resolve to one IRI each                                                                                                                              | All eight IRIs resolve; else use `type_ancestors`                                                                                             |
| B:25 to B:31 | 6   | Blocking and candidate pairs | `silver_candidate_pairs` with DOUBLE-cast features and `label`                                                                                                                                                                                          | **B:31: pairs table exists, label has both classes**                                                                                          |
| B:31 to B:41 | 10  | Model                        | Train with `GroupShuffleSplit` on `obs_id`; print metrics; score in memory; `silver_model_decisions`; band counts and precision per band. Then attempt MLflow log (cap 3 minutes, see stall rule)                                                       | **B:41: band counts printed.** Measured: REVIEW is 996 rows (22.3%), accepted as a ranked queue. Do not tune seeds or thresholds to shrink it |
| B:41 to B:45 | 4   | Nominations                  | Edge table, label propagation loop with an `assert` on convergence, `gold_nominations`; print `noms` and `recovered`                                                                                                                                    | Hero C found                                                                                                                                  |
| B:45 to B:49 | 4   | MERGE and approval           | MERGE `AUTO` counts and nominations into `oms_objects`; approval cell for hero B (writes `review_decisions`, MERGEs the decision)                                                                                                                       | Object count before and after printed                                                                                                         |
| B:49 to B:54 | 5   | Dossier                      | `object_dossiers` for A and B using retrieved rows only; bracketed IDs; ID-existence check                                                                                                                                                              | Dossier prints with valid citations                                                                                                           |
| B:54 to B:60 | 6   | Presentation                 | Deploy the App from `app/`; check it lists the review queue and writes a decision. Fallback: AI/BI dashboard on `oms_objects`, `silver_model_decisions`, `review_queue` (point map, queue table, counts). Note hero IDs and metrics into the cheat card | **B:60: MVB complete.** Anything unfinished is cut, not extended                                                                              |

### Checkpoint rules during the build

| Build minute | Must be true                      | If not                                                                                                                            |
| ------------ | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| B:20         | Data landed with seeds            | Spend at most 5 more minutes; then ship the generator without the least valuable seed (noise, then marking mix)                   |
| B:31         | Candidate pairs exist             | Rounded-grid blocking fallback                                                                                                    |
| B:41         | Decisions and band counts printed | Weighted-score rule decision (see fallback table), with ML presented as the trained in-memory model and registration as next step |
| B:49         | MERGE done                        | `INSERT OVERWRITE` an outbox table                                                                                                |
| B:54         | Dossier prints                    | Template dossier (no LLM), show Genie only if time                                                                                |
| B:60         | App or dashboard exists           | Notebook `display` of the same tables plus a folium map                                                                           |

## 2. Presentation prep

| Step        | Do                                                                                                                                                                | Output                |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| Run all     | **Run all from the top** (the reset cell makes it safe). Note any cell that errors                                                                                | Pass/fail for run-all |
| Cheat card  | Hero IDs (A, B, C), the printed counts, the metrics, the working `ai_query` endpoint, the three threshold numbers. Every number labelled **synthetic**            | One page              |
| Genie       | Genie space on gold tables. Test two questions: object count by type, and observations awaiting review (cut line 2 if not ready)                                  | Genie working or cut  |
| Backups     | Screenshots of the dashboard, the review queue, the dossier, the lineage view (if it works), the band counts; open all tabs in demo order                         | Screenshot folder     |
| Timed run   | Run file 05 end to end: slides plus live notebook plus dashboard, speaking aloud, with the clock visible. Note every point where speech exceeds its minute budget | List of rough spots   |
| Final check | Warehouse running, notebook fully run and outputs visible, tabs in order, notifications off, screenshots reachable                                                | Ready                 |

Fix rough spots by changing what is said, not what is run. If the build ran over, drop Genie first, then shrink the backups to three screenshots. The timed run is never dropped.

## 3. Objection drill

Practise one line each on cost, complexity, trust in the data, AI reliability, security/ATO and competition (file 07). Flag any authorization claim as "I will confirm in writing".

## 4. Cut lines (in this order)

| Order           | Cut                                                                                                                      | What I say instead                                                                                                                                                  | Cost                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 1               | **Genie space**                                                                                                          | Show the same question as a SQL query over `oms_objects`; say Genie runs on the same governed tables                                                                | Loses the plain-English moment                     |
| 2               | **UC model registration** (keep MLflow logging if it works; otherwise keep the in-memory model)                          | "Registration in Unity Catalog is the governed next step; the run and metrics are tracked." If logging worked but registration failed, load from `runs:/<id>/model` | Weaker governance story for Layer 2                |
| 3               | **App deployment** (switch to the fallback after 10 minutes stalled; the notebook approval cell already closes the loop) | "The analyst screen is the Databricks App in `app/`; today it is a notebook approval cell and dashboard."                                                           | Less polished UI                                   |
| 4               | **Dashboard polish** (colours, extra tiles, filters)                                                                     | Basic point map, queue table, counts                                                                                                                                | Looks plain                                        |
| 5 (last resort) | **Simplify clustering** via the pandas/networkx fallback on the small unmatched set                                      | Say honestly that the production path is distributed label propagation                                                                                              | Weakens the "Spark SQL, no extra dependency" claim |
| Never           | **Any layer** (data, model, GenAI, presentation)                                                                         | Thin version of each: template dossier, weighted-score rule plus trained in-memory model, notebook presentation                                                     |                                                    |

## 5. Stall rule

Per AGENTS.md: if a step stalls more than 10 minutes, go to its fallback.

1. **Minutes 0 to 3:** read the full error, make one targeted fix.
2. **Minute 3:** write one line in a `stall_log` markdown cell (what, when, what I tried). This feeds the retrospective.
3. **Minute 3 to 10:** one more attempt at most, using a different approach, not the same fix again.
4. **Minute 10:** switch to the fallback below, move on, and do not return unless MVB is complete with time left.
5. A step that **overruns its budget by half** (for example 15-minute data step at 22 minutes) triggers a checkpoint decision: cut a seed, not a layer.

| Step              | Fallback                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `COPY INTO`       | `spark.read.json` on the volume plus append                                                                                                                        |
| CCO flatten       | Hand-built `type_ancestors` table                                                                                                                                  |
| H3 functions      | Rounded lat/lon grid cells plus neighbours                                                                                                                         |
| MLflow log        | Skip logging; keep the in-memory model; log later if time                                                                                                          |
| UC registration   | Load from `runs:/<id>/model`                                                                                                                                       |
| Model training    | A weighted-score rule (distance plus type score) with the same three-way decision; present ML as pilot scope (last resort, this cuts a layer to its thinnest form) |
| Label propagation | pandas/networkx on the small unmatched set                                                                                                                         |
| MERGE             | `INSERT OVERWRITE` an outbox table                                                                                                                                 |
| `ai_query`        | Template-based dossier; Genie for Q&A                                                                                                                              |
| Row filter        | Secure view with `is_account_group_member()`                                                                                                                       |
| Map               | folium via `displayHTML`, or scatter of lat/lon                                                                                                                    |
| Databricks App    | Notebook approval cell plus AI/BI dashboard                                                                                                                        |
| AI/BI dashboard   | Notebook `display` plus folium                                                                                                                                     |

## 6. Things not to do (hard rules)

| Do not                                                                                        | Why                                                                                       |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Reintroduce **Auto Loader**                                                                   | Hung on Free Edition; decision made. `COPY INTO` only                                     |
| Reintroduce **Splink**                                                                        | Decision made; pairs are observation-to-object                                            |
| Reintroduce **GraphFrames**                                                                   | Decision made; label propagation in Spark SQL                                             |
| Reintroduce **Vector Search**                                                                 | Decision made; the dossier is a structured join                                           |
| Add **computer vision**                                                                       | Decision made                                                                             |
| Add model features without updating file 03                                                   | Feature list is fixed: `dist_m`, `type_score`, `type_missing`, `confidence`, `is_analyst` |
| Use `true_object_id` as a feature or in the dossier prompt                                    | Leakage; it stands in for past adjudications only                                         |
| Tune thresholds or retrain to improve metrics                                                 | Report honest synthetic numbers; thresholds are demo values                               |
| Claim federation to real Postgres/MySQL, an ATO, a FedRAMP or IL level, or that a retrain ran | Not tested or not verified                                                                |
| Add events, real data, real schemas or anything classified-looking                            | Out of scope; synthetic data only                                                         |
| Add a dependency not already in `pyproject.toml` or the pip cell                              | Complexity and restart risk                                                               |
| Polish before the MVB is complete                                                             | Presentation polish is cut line 4                                                         |
| Debug a single error past 10 minutes                                                          | Stall rule                                                                                |
| Invent a statistic or baseline                                                                | Ask the audience; use placeholders                                                        |

## 7. Checks before a demo

### 7a. Known pitfalls (already handled in the notebooks)

| #   | Pitfall                                                                                                                                                                                          | Handling                                                                                                                                                                       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Spark `CASE` with decimal literals returns `DECIMAL`; pandas then holds `Decimal` objects and `mlflow.sklearn.log_model` fails with `TypeError: Object of type Decimal is not JSON serializable` | `CAST(... AS DOUBLE)` in SQL and `astype("float64")` after `toPandas()`                                                                                                        |
| 2   | Registration cell fails as a cascade when logging failed (`NameError: RUN_ID`)                                                                                                                   | Guard registration so it only runs when `RUN_ID` exists                                                                                                                        |
| 3   | Unity Catalog registration needs a model signature                                                                                                                                               | Pass an explicit signature (`mlflow.set_registry_uri("databricks-uc")`)                                                                                                        |
| 4   | Type labels not found in CCO (e.g., Airfield, Radar Site)                                                                                                                                        | CCO-native leaf types; `type_score` from `cco_classes.ancestor_iris` (file 03, section 1)                                                                                      |
| 5   | Markings that look like real banners                                                                                                                                                             | `OPEN` / `RESTRICTED` in a column named `marking`                                                                                                                              |
| 6   | An empty REVIEW queue                                                                                                                                                                            | Seed twins (about 25 pairs), duplicate-object evidence (5 twins), partial observations (about 30 nominated objects, 3 each), noise (about 5%) and cross-producer corroboration |
| 7   | Label propagation loop breaking silently at the iteration cap                                                                                                                                    | `assert` that it converged before the 14-iteration cap; run it over `silver_model_decisions` (`NOMINATE`)                                                                      |
| 8   | Unknown `ai_query` endpoint                                                                                                                                                                      | Probe the endpoint list in the feasibility notebook; the working one is `databricks-meta-llama-3-3-70b-instruct`                                                               |

### 7b. Manual checks

| #   | Check                                                                                                | If it fails                                                                     |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | **Genie** space on gold tables answers an object-count question                                      | Cut line 2 stays; rehearse the SQL substitute                                   |
| 2   | **Databricks Apps**: `app/` deploys, queries through the SQL warehouse and writes `review_decisions` | Use the notebook approval cell and AI/BI dashboard; update the script (file 05) |
| 3   | **Lineage graph** in Catalog Explorer shows bronze → silver → gold → OMS                             | Describe lineage verbally; use screenshot only if real                          |
| 4   | **AI/BI dashboard point map** renders lat/lon                                                        | Notebook folium map is the fallback                                             |
| 5   | **MLflow UI** shows the run, metrics, and (if registered) the UC model                               | See 7a items 2 and 3                                                            |

### 7c. Record and decide

| #   | Item                                                                                                                                                                                  | Output                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | Paste **model metrics** (AUC, average precision) and **precision per decision band** vs `true_object_id`, all labelled synthetic                                                      | Numbers for files 05 and 06 (current values are in file 02). If AUTO precision is poor, fix the generator, not the story |
| 2   | **Time the whole pipeline**, then run **timed from-scratch build drills** (blank notebook, only the pipeline notebook open for reference). If a drill exceeds 60 minutes, cut earlier | Measured typing time, not just compute time (compute is minutes, so typing is the risk)                                  |
| 3   | Count `AUTO` / `REVIEW` / `NOMINATE`, nominations and recovered observations                                                                                                          | Replace every `[fill from run, synthetic]` placeholder                                                                   |
| 4   | Pick hero IDs A, B, C and write the cheat card                                                                                                                                        | One page                                                                                                                 |
| 5   | Verify public claims (DoD/IC baseline wording; any authorization level) from official sources                                                                                         | Exact wording on a card, or "I will confirm in writing"                                                                  |

## 8. One-card summary

- Order: data → model → GenAI → presentation. Never cut a layer.
- Cut order: Genie, UC registration, App (switch to dashboard fallback), dashboard polish, then clustering fallback.
- Thresholds: `AUTO` ≥ 0.9, `REVIEW` 0.5 to 0.9, `NOMINATE` < 0.5. Features all DOUBLE.
- Heroes: A auto (Armored Fighting Vehicle), B twin Trucks (review), C nominated cluster.
- Stall: 3 minutes to one fix, 10 minutes to fallback.
- No Auto Loader, Splink, GraphFrames, Vector Search or CV.
