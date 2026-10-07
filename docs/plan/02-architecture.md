# 02: Architecture

Purpose: components and data flow across the four layers, the trade-offs to defend, and what is mocked versus production. Table names and thresholds come from file 03.

## Shared facts (every other deliverable must agree with these)

| Item                    | Value                                                                                                                                                                                                                                                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Core entity             | Object. Build covers facility and equipment types; events are out of scope.                                                                                                                                                                                                                                                                      |
| Leaf object types (CCO) | Airport, Military Facility, Truck, Armored Fighting Vehicle, Aircraft (IRIs in file 03)                                                                                                                                                                                                                                                          |
| Decision thresholds     | `AUTO` ≥ 0.9, `REVIEW` 0.5–0.9, `NOMINATE` < 0.5 or no candidate                                                                                                                                                                                                                                                                                 |
| Model                   | scikit-learn `HistGradientBoostingClassifier`; features `dist_m`, `type_score`, `type_missing`, `confidence`, `is_analyst`, `n_candidates`, `dist_gap_m`, `days_since_last_seen` (all cast to DOUBLE; definitions in 03-data-model)                                                                                                                                                                                                  |
| Blocking                | H3 resolution 8 with `h3_kring` radius 1; candidate radius 500 m                                                                                                                                                                                                                                                                                 |
| Clustering              | Iterative min-label propagation in Spark SQL, clusters of ≥ 2 observations become nominations                                                                                                                                                                                                                                                    |
| Ingestion               | `COPY INTO` (incremental, idempotent). Auto Loader hung on Free Edition; streaming is the production path only                                                                                                                                                                                                                                   |
| Hero objects            | A = clean auto-associate (Armored Fighting Vehicle), B = twin-Truck review case, C = nominated cluster of partial observations                                                                                                                                                                                                                   |
| Presentation            | Primary: Databricks App (`app/app.py`: review queue, candidate map, dossier, approve/reject write-back). Fallback: notebook walkthrough + AI/BI dashboard + Genie (example queries in `demo/genie-examples.md`). Nothing is deployed: the app, dashboard and Genie space are created in the target workspace after the pipeline notebook has run |
| Data                    | Synthetic. Every demo number is labelled synthetic. Measured band counts and precision are in the block below                                                                                                                                                                                                                                    |

### Platform feasibility
`notebooks/feasibility_tests.py` checks each platform dependency in an independent block (summary table at the end). Compute time is not the constraint; typing and debugging time is. All 18 automated checks passed on Databricks Free Edition serverless (Python 3.12, Spark 4.2.0, scikit-learn 1.7.2, mlflow-skinny 3.12.0, rdflib 7.6.0, folium 0.20.0), including Unity Catalog model registration and the full CCO file parse. Working `ai_query` endpoint: `databricks-meta-llama-3-3-70b-instruct`. Genie, Databricks Apps, the lineage graph and the AI/BI point map are manual checks (feasibility notebook, section 12) not yet done.

### Pipeline notebook run (`notebooks/generate_data.py`): the numbers to use in the demo
Synthetic data. Band precision is **in-sample**; AUC and average precision are **held-out** (grouped by observation).

| Item                       | Value                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Observations / OMS objects | 4,474 / 2,025 seeded; 2,171 after 146 nominations are merged                                                                           |
| AUTO                       | 2,981 (66.6%), precision 0.989                                                                                                         |
| REVIEW                     | 810 (18.1%), precision 0.962 (a ranked queue; not tuned)                                                                               |
| NOMINATE                   | 683 (15.3%), precision 0.933                                                                                                           |
| Held-out                   | AUC 0.859, average precision 0.937                                                                                                     |
| Nominations / recovered    | 146 nominations, 352 observations recovered (7.9% of observations)                                                                     |
| Seeded duplicates surfaced | 3 of 5                                                                                                                                 |
| Model                      | `workspace.obj_resolution_demo.obj_resolution_match_model` v1 (Unity Catalog)                                                          |
| Heroes                     | A `OBJ-00599`; B `OBJ-T005` (obs `OBS-559829602922`, look-alike `OBJ-00081`, 0.854 vs 0.304); C `NOM-406859eed6` (4 weak observations) |
| Compute                    | 4.8 minutes end to end                                                                                                                 |

No rules baseline has been run, so make no claim of an edge over rules. Caveat: hero B's runner-up is 0.304, so it is a REVIEW case because it falls below 0.9, not because the two candidates are tied.

## 1. Data flow (text diagram)

```
LAYER 1  GOVERNED DATA FOUNDATION  (Unity Catalog: one catalog.schema, volumes, lineage, row filter)

 [Postgres OMS]*   [MySQL tasking]*      [SaaS/API observation JSON]*   [Report text]*      [CCO .ttl (public)]
   oms_objects       landing/tasking/*.csv   landing/observations/*.json    landing/reports/*   data/ontology/
        |                 |  COPY INTO           |  COPY INTO                   | COPY INTO          | rdflib parse
        |                 v                      v                              v                    v
        |           bronze_tasking        bronze_observations           bronze_reports          cco_classes
        |           (source_file, ingested_at on every bronze table = lineage + audit)         (ancestor_iris)
        |                                        |                                                  |
LAYER 2  PREDICTIVE ML                           v                                                  |
        |                  H3 blocking (res 8, kring 1, <= 500 m) ---- type compatibility <---------+
        |                                        v
        |                            silver_candidate_pairs  (dist_m, type_score, type_missing, confidence,
        |                                        |            is_analyst, n_candidates, dist_gap_m,
        |                                        |            days_since_last_seen, label)
        |                      scikit-learn gradient boosting -> MLflow run -> UC model obj_resolution_match_model**
        |                                        v
        |                            silver_model_decisions  (match_prob)
        |                          /             |                     \
        |                  AUTO >= 0.9     REVIEW 0.5-0.9        NOMINATE < 0.5 / no candidate
        |                      |                 |                          |
        |                      |            review_queue              label propagation
        |                      |                 |  analyst approve/reject    (Spark SQL)
        |                      |                 v                          v
        |                      |          review_decisions        gold_nominations (>= 2 obs)
        |                      |                 |  -> new training labels (retraining loop)
        v                      v                 v                          v
   oms_objects  <------------------------- MERGE write-back ----------------------------+
                                                 |
LAYER 3  GenAI                                   v
                         object dossier: ai_query over [observations + match_prob + bronze_reports]
                         for one object, cites obs_id / report_id          Genie space on gold tables
                                                 |
LAYER 4  PRESENTATION                            v
        map (folium / point map) -> select object -> match probability and why -> dossier
        -> approve / reject (MERGE).   Primary: Databricks App.  Fallback: notebook + AI/BI dashboard + Genie.

 *  mocked source (synthetic)             ** registration checked by the feasibility notebook
```

## 2. Layer by layer

### Layer 1: governed data foundation
- **What it does:** lands four source types onto one entity (the Object) with `source_file` and `ingested_at` on every row, aligns types to CCO via `cco_classes`, and protects rows with a Unity Catalog row filter on `marking`.
- **Proven on Free Edition:** schema and volume, `COPY INTO` incremental and idempotent, rdflib flatten, row filter DDL.
- **Mocked:** Postgres and MySQL are Delta tables and CSV files. Production path: Lakehouse Federation or CDC from the real databases. **Untested on Free Edition. Do not claim it ran.**
- **Production talking points, not built:** lineage (Catalog Explorer, to be confirmed in the manual check), data-quality expectations on bronze, attribute-based access control, audit logs.
- **Row filter limit:** a single-user workspace cannot show two identities seeing different rows. Show the function, the `marking` column and the Catalog Explorer view; say it is a single-user demo.

### Layer 2: predictive ML
- **What it does:** scores every observation-to-object candidate pair with a match probability and turns it into one of three actions.
- **Why a classifier:** it replaces hand-tuned weights (a simple weighted-score rule is the fallback, see file 04) with a model trained on past adjudications, so the system learns from outcomes.
- **Monitoring and retraining (talk track):** log the run in MLflow; monitor the `AUTO` share, `REVIEW` volume and the reject rate in the review queue; `review_decisions` become labels for the next training run.
- **Honest limit:** the training labels come from `true_object_id`, which simulates adjudications. The measured precision is on synthetic data, is in-sample, and says nothing about real-world accuracy. Say so.

### Layer 3: GenAI activation
- **What it does:** builds a short dossier for one object from governed tables (observations, match probabilities, report text) and cites the IDs it used. A Genie space answers plain-English count questions over gold tables.
- **Reliability controls:** the prompt restricts the model to the retrieved rows ("using ONLY the reports below"), requires bracketed IDs, and the demo checks that cited IDs exist in the retrieved set. The dossier never changes object data; the analyst still approves.
- **Free Edition:** `ai_query` is checked by the feasibility notebook, whose working endpoint is `databricks-meta-llama-3-3-70b-instruct`.
- **Production caveat (sourced in file 07, section 5, F3/F4):** the Databricks GovCloud docs list AI Functions, which includes `ai_query`, as unavailable on AWS GovCloud, and IL5 model availability is only partially confirmed. Say "the dossier pattern is portable; in a government environment the call goes to a model serving endpoint that is available there, and I would confirm which with the account team." Do **not** say `ai_query` runs at IL5. Retrieved 2026-10-02 via a summarizer; open the primary pages before relying on it.

### Layer 4: presentation
- **Primary:** a Databricks App with one screen: review queue → select item → probability and why → candidate and runner-up on a map → dossier → approve/reject (write-back). Deployment steps (SQL warehouse resource, service-principal grants) are in the `app/app.py` docstring; the pipeline notebook must have run first. Not deployed until you do it.
- **Fallback:** a notebook flow (including the approval cell) plus an AI/BI dashboard (point map, review queue table, counts) and a Genie space. These need no app deployment.
- **Fallback for the map:** folium through `displayHTML` (works in the pipeline notebook).

## 3. Trade-offs to defend

| Decision                                                       | Why                                                                                                                         | What I give up                                                                            | Production path                                                                                                                         |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `COPY INTO` instead of Auto Loader                             | Idempotent, no streaming query, no checkpoint; Auto Loader hung on Free Edition                                             | Not continuous                                                                            | Auto Loader / Lakeflow with schema evolution and file notification                                                                      |
| Gradient boosting classifier instead of rules                  | Learns from adjudications; predicted probabilities drive thresholds (not guaranteed calibrated; calibration is a next step) | Needs labelled pairs; harder to explain than a formula                                    | Build a rules baseline to compare against, and explain the model with feature importance and the top contributing features per decision |
| Classifier instead of Splink                                   | One fewer dependency; the pairs are observation-to-object, not record-to-record; decision already made                      | Splink's probabilistic linkage tooling                                                    | Revisit if object-to-object deduplication becomes the main problem                                                                      |
| H3 blocking (res 8, kring 1)                                   | Cuts the pair space; native SQL; runs in the pipeline notebook                                                              | Misses matches beyond the ring; resolution is a tuning choice                             | Tune the resolution per object class and positional error                                                                               |
| Label propagation in Spark SQL instead of GraphFrames          | No extra dependency; runs in the pipeline notebook                                                                          | More iterations at scale; fiddly                                                          | GraphFrames or a connected-components library at volume                                                                                 |
| `ai_query` over governed tables instead of RAG / Vector Search | The dossier is a structured join, not semantic search; decision already made                                                | No semantic retrieval over a large report corpus                                          | Add Vector Search over the report text when the corpus is large                                                                         |
| CCO via flattened Delta table instead of a graph store         | Type compatibility is an ancestor lookup, so a table is enough                                                              | No reasoning or SPARQL                                                                    | Graph/triple store only if inference is needed                                                                                          |
| Thresholds 0.9 and 0.5                                         | Auto-associate only when very sure, so analysts only see the uncertain middle                                               | Thresholds are demo values, not tuned                                                     | Tune with analysts against a cost of false merge versus false split                                                                     |
| App primary, notebook + dashboard fallback                     | The App is the analyst's real workflow and closes the write-back loop; the fallback needs no app deployment                 | Deployment dependencies (warehouse resource, grants); switch to the fallback if it stalls | App with monitoring and row-level policies in production                                                                                |

## 4. Free Edition limits and flags

| Area                                       | Status                                                                        | Handling                                                                                             |
| ------------------------------------------ | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Auto Loader                                | Known to hang (hard 5-minute timeout in the notebook)                         | Not used                                                                                             |
| MLflow model logging                       | Needs DOUBLE-cast features (feasibility notebook, section 5)                  | Keep the DOUBLE cast rule                                                                            |
| UC model registration                      | Checked by the feasibility notebook                                           | If it fails live, load from `runs:/<id>/model`                                                       |
| Databricks Apps                            | Not deployed; deployment checked manually (feasibility notebook, section 12)  | Notebook approval cell + AI/BI dashboard                                                             |
| Genie                                      | Unverified                                                                    | Verify before the demo; dashboard fallback                                                           |
| Lineage graph                              | Unverified                                                                    | Verify before the demo; otherwise describe it                                                        |
| Row filter with multiple identities        | Not demonstrable on a single user                                             | Show function and column                                                                             |
| Federation to real Postgres/MySQL          | Not tested                                                                    | Mocked; say so                                                                                       |
| Foundation Model endpoints and rate limits | `ai_query` checked; working endpoint `databricks-meta-llama-3-3-70b-instruct` | Re-run the feasibility notebook if the endpoint changes; have a template-based dossier as a fallback |

## 5. Public claims to verify before they are said out loud
| Claim                                                           | Status                                                                                                                      |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| BFO and CCO are DoD/IC baseline ontology standards "since 2024" | Partially confirmed (university and CUBRC releases, no primary DoD/IC document). Use the safe wording in file 07, section 7 |
| CCO release used                                                | The local file's version IRI is `.../2026-08-13/CommonCoreOntologiesMerged` (verified in the file)                          |
