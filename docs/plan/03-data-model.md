# 03: Data model

Purpose: the canonical tables, CCO classes and seeded cases the live build must produce, so the demo shows recovered data, a real review queue and a sourced dossier. This file is the source of truth for table names, thresholds and hero objects; files 01, 02 and 04–07 must match it.

All data is synthetic. Every number in this file marked `~` is an expected value from the notebook's generator parameters, **not** a measured result. Replace with measured values after a notebook run.

## 1. Type taxonomy decision

Object types are **CCO-native labels**, verified against `data/ontology/CommonCoreOntologiesMerged.ttl` (version IRI `https://www.commoncoreontologies.org/2026-08-13/CommonCoreOntologiesMerged`). Type compatibility is computed from `cco_classes.ancestor_iris` (the rdflib flatten), not from a hand-built taxonomy, so "ontology-conformant" is a true claim.

Where CCO's label differs from the everyday name:

| Everyday name                                         | CCO label used             | Note                                                                                                                            |
| ----------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Airfield                                              | `Airport`                  |                                                                                                                                 |
| Radar site                                            | `Military Facility`        | CCO has only radar *imaging artifact functions*, no radar-site class                                                            |
| Fixed-wing aircraft                                   | `Aircraft`                 | No fixed-wing subclass                                                                                                          |
| Armored vehicle                                       | `Armored Fighting Vehicle` |                                                                                                                                 |
| Artifact (generic root)                               | `Material Artifact`        |                                                                                                                                 |
| Facility / Ground Vehicle / Vehicle (generic parents) | same labels                | In CCO, `Aircraft` is a child of `Vehicle`, not a sibling-parent, so ancestors are derived from `cco_classes`, never hand-built |

A small hand-built `type_ancestors` table is the fallback only if the flatten breaks.

Test: the pipeline notebook asserts that all 5 leaves and 3 generics resolve to exactly one IRI each.

## 2. Chosen CCO / BFO classes

IRIs verified by grep against the local `.ttl`. CCO IRIs have the form `https://www.commoncoreontologies.org/<id>`; BFO IRIs have the form `http://purl.obolibrary.org/obo/BFO_<id>`.

### Object types (the core entity)
| Role in demo                              | CCO label                | IRI                                                | Parent chain (verified)                                                                                                              |
| ----------------------------------------- | ------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Leaf: facility                            | Airport                  | `https://www.commoncoreontologies.org/ont00001078` | Transportation Facility `ont00000226` → Facility `ont00000192` → Material Artifact `ont00000995` → BFO material entity `BFO_0000040` |
| Leaf: facility                            | Military Facility        | `https://www.commoncoreontologies.org/ont00001052` | Facility `ont00000192` → Material Artifact                                                                                           |
| Leaf: equipment                           | Truck                    | `https://www.commoncoreontologies.org/ont00000606` | Ground Motor Vehicle `ont00000053` → Ground Vehicle `ont00000618` → Vehicle `ont00000713` → Material Artifact                        |
| Leaf: equipment                           | Armored Fighting Vehicle | `https://www.commoncoreontologies.org/ont00000427` | Ground Motor Vehicle `ont00000053` → Ground Vehicle → Vehicle                                                                        |
| Leaf: equipment                           | Aircraft                 | `https://www.commoncoreontologies.org/ont00001043` | Vehicle `ont00000713` → Material Artifact                                                                                            |
| Generic (what a weak observation reports) | Facility                 | `https://www.commoncoreontologies.org/ont00000192` |                                                                                                                                      |
| Generic                                   | Ground Vehicle           | `https://www.commoncoreontologies.org/ont00000618` |                                                                                                                                      |
| Generic                                   | Vehicle                  | `https://www.commoncoreontologies.org/ont00000713` |                                                                                                                                      |

### Supporting classes (data-model alignment, not used as match types)
| Use                             | CCO label          | IRI                                                |
| ------------------------------- | ------------------ | -------------------------------------------------- |
| An observation record           | Act of Observation | `https://www.commoncoreontologies.org/ont00000037` |
| Finished report text            | Report             | `https://www.commoncoreontologies.org/ont00002043` |
| Collection sensor               | Sensor             | `https://www.commoncoreontologies.org/ont00000569` |
| Area of interest / tasking area | Geospatial Region  | `https://www.commoncoreontologies.org/ont00000472` |

### Not in scope for the build
- **Events** (the core entity is "facility, equipment, event"): out of scope for the initial build. Say so, and say the same pipeline applies to events with BFO `process` / CCO `Act` classes. I did not verify specific event IRIs, so do not cite any.
- Equipment generic **"Ground Motor Vehicle"** (`ont00000053`) is also a valid generic report type. Include it only if time allows.

### What the type compatibility rule does with these classes
- Observation type = object type → `type_score` 1.0
- Observation type is an ancestor of the object type (for example `Vehicle` vs `Truck`) → 0.8, "compatible but generic"
- Observation type missing → 0.5, and `type_missing = 1` as its own feature
- Otherwise (siblings or unrelated, for example `Military Facility` vs `Airport`) → 0.0, which is blocked from the candidate set
- **Deliberate ambiguity:** a generic `Vehicle` observation is an ancestor of Truck, Armored Fighting Vehicle *and* Aircraft, so it matches all three at 0.8. This is a feature, not a bug. It generates real REVIEW cases.

Illustrative only (not code to commit): `type_score` comes from `array_contains(cco.ancestor_iris, obs_type_iri)` joined on the object's type IRI. All type scores must be cast to DOUBLE (see the Decimal/MLflow note in file 04).

## 3. Tables

Naming follows the pipeline notebook (`<catalog>.<schema>.<table>`; schema `obj_resolution_demo`). Layers: bronze = landed as-is plus lineage columns, silver = joined and scored, gold = what the analyst and the object system consume.

### Layer 1: sources (mocked) and bronze
| Table / file                                                | Mocks                                                     | Source type       | Key columns                                                                                                                                                                                              | Ingest                                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `oms_objects`                                               | Postgres object management system (authoritative objects) | Relational        | `object_id` PK, `object_type` (CCO label), `object_type_iri`, `lat`, `lon`, `marking`, `first_seen`, `last_seen`, `obs_count`, `source`, `status`                                                        | Written directly as Delta; production path is Lakehouse Federation or CDC (**untested on Free Edition; do not claim it ran**) |
| `landing/tasking/*.csv` → `bronze_tasking`                  | MySQL collection/tasking records                          | Relational export | `task_id`, `sensor`, `area_wkt` or bbox, `start_time`, `end_time`, `requested_by`, `priority`, `source_file`, `ingested_at`                                                                              | `COPY INTO` (CSV)                                                                                                             |
| `landing/observations/batch_*.json` → `bronze_observations` | SaaS/API algorithm and analyst feeds                      | JSON API export   | `obs_id` PK, `obs_time`, `lat`, `lon`, `reported_type`, `confidence`, `producer` (`algorithm`/`analyst`), `sensor` (`EO`/`SAR`/`FMV`), `source_file`, `ingested_at`, **`true_object_id` (scoring only)** | `COPY INTO` (incremental and idempotent)                                                                                      |
| `landing/reports/*.json` → `bronze_reports`                 | Finished report text                                      | Documents         | `report_id`, `object_id` (nullable), `obs_id` (nullable), `report_text`, `report_time`, `source_file`, `ingested_at`                                                                                     | `COPY INTO`                                                                                                                   |
| `cco_classes`                                               | CCO via rdflib                                            | Ontology          | `iri`, `label`, `parent_iris`, `ancestor_iris`                                                                                                                                                           | rdflib flatten (runs in the pipeline notebook)                                                                                |

- Two required source *types* are met by relational (OMS + tasking) plus JSON feed. Reports add a document source.
- `true_object_id` is **never** a model feature. It stands in for past analyst adjudications when generating labels, and for scoring the demo. Say this out loud in the demo. In production the labels come from the review queue (`review_decisions`).
- `marking` values are `OPEN` / `RESTRICTED`, so nobody mistakes them for real classification banners.

### Layer 2: silver
| Table                                              | Built from                                                                 | Key columns                                                                                                                                 |
| -------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `silver_candidate_pairs`                           | H3 blocking (res 8, `h3_kring` 1) on `bronze_observations` × `oms_objects` | `obs_id`, `object_id`, `dist_m`, `type_score`, `type_missing`, `confidence`, `is_analyst`, `label`                                          |
| `silver_model_decisions`                           | Model scoring, best candidate per observation                              | `obs_id`, `matched_object_id`, `match_prob`, `decision` (`AUTO` ≥ 0.9, `REVIEW` 0.5–0.9, `NOMINATE` < 0.5 or no candidate), `model_version` |
| MLflow run + UC model `obj_resolution_match_model` | `HistGradientBoostingClassifier` on `silver_candidate_pairs`               | Registration needs DOUBLE-cast features and an explicit signature                                                                           |

Features (all DOUBLE): `dist_m`, `type_score`, `type_missing`, `confidence`, `is_analyst`. Do not add features without updating this spec.

### Layer 2 → 3: gold
| Table                                                                             | Built from                                                                     | Key columns                                                               |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `nom_edges`, `cc_0`, `cc_1`, `cc_final`                                           | Label propagation over `NOMINATE` observations (runs in the pipeline notebook) | `src`/`dst`, `id`/`label`                                                 |
| `gold_nominations`                                                                | Clusters with ≥ 2 observations                                                 | `object_id` (`NOM-…`), `object_type`, `lat`, `lon`, `obs_count`           |
| `review_queue` (a view over `silver_model_decisions` where `decision = 'REVIEW'`) |                                                                                | `obs_id`, `candidate object_id`, `match_prob`, `reason`                   |
| `review_decisions`                                                                | Analyst approve/reject (new, written live)                                     | `obs_id`, `object_id`, `decision`, `decided_by`, `decided_at`             |
| `oms_objects` (MERGE target)                                                      | `AUTO` associations plus nominations plus approved reviews                     | adds rows with `source = 'DATABRICKS_NOMINATION'`, `status = 'NOMINATED'` |
| `silver_reports` / `object_dossiers`                                              | `ai_query` over observations, scores and `bronze_reports` for one object       | `object_id`, `dossier_text`, `cited_ids`, `generated_at`                  |

`review_decisions` is the retraining loop: approved and rejected pairs become new `label` values for `silver_candidate_pairs` on the next training run. In the demo, show the write; describe the retrain as the next step. Do not claim a retrain ran unless it did.

## 4. Generator parameters and seeded cases

### Base volumes (from the notebook; expected, not measured)
| Parameter                              | Value                                                                         | Expected result     |
| -------------------------------------- | ----------------------------------------------------------------------------- | ------------------- |
| OMS objects                            | 2,000 uniform over bbox 39.0–39.5 N, 105.5–105.0 W (arbitrary synthetic area) | ~2,000              |
| Objects with observations              | 1,500, each with 1–4 observations (mean 2.5)                                  | ~3,750 observations |
| Not in OMS (should become nominations) | 150 new objects, 2–3 observations each                                        | ~375 observations   |
| Observation noise                      | position σ 80 m (algorithm), 150 m (analyst)                                  |                     |
| Reported type                          | 70% exact, 20% generic parent, 10% missing                                    |                     |
| Producer mix / sensors                 | 70% algorithm, 30% analyst; EO, SAR, FMV                                      |                     |
| Time window                            | last 72 hours                                                                 |                     |

Expected ≈ 4,100 observations; **measured 4,474** (section 6). Confirm the printed `objects=…` line (expected 2,025 = 2,000 plus 25 twins).

### Seeded cases (add deliberately)
The uniform generator makes clean cases easy and REVIEW cases rare, which would leave the review queue nearly empty. Add these, then check the counts.

1. **Twin objects (about 25 pairs).** Two OMS records 150–400 m apart with compatible types (for example two Trucks, or an Armored Fighting Vehicle and a Truck). Observations placed between them with generic `Vehicle` or missing type. This is the "how many objects really are there?" case, and it produces genuine `REVIEW` rows.
2. **Duplicate-object evidence.** For 5 of the twins, give the same underlying `true_object_id` to observations pointing at both records. This lets you say "the model surfaced that these two records may be one object", and lets the object count move before and after.
3. **Partial single-source observations (the "discarded data" story).** For each of about 30 of the `NEW-` objects, generate 3 observations that are individually unusable: one missing type, one generic type, one specific type but low confidence (0.4–0.55) and one km-scale position error. No single one passes the `AUTO` threshold or can support an analyst judgment alone. Together they cluster into one nomination.
4. **Low-confidence algorithm noise.** About 5% of observations with no real object behind them (`true_object_id = NULL`), single and isolated. These must stay `NOMINATE` and **not** cluster, to show the pipeline does not over-merge.
5. **Cross-producer corroboration.** For some existing objects, add one algorithm and one analyst observation within 100 m. These are the clean `AUTO` cases.
6. **Marking mix.** Keep ~20% of OMS objects `RESTRICTED` (demo row filter).

### Hero objects (named, memorised, used in the demo)
Pick IDs after a run, and keep a one-line cheat card.
| Hero                         | Scenario                                                                                 | Shows                                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **A: clean auto-associate**  | Armored Fighting Vehicle, 4+ observations (algorithm + analyst), match_prob > 0.95       | Auto-associate, map, dossier with citations                                                               |
| **B: review**                | Twin Trucks 200 m apart; a generic `Vehicle` observation with `match_prob` ~0.6 for each | Review queue, "why this probability", approve/reject, MERGE write-back, label for retraining              |
| **C: recovered / nominated** | `NEW-` cluster of 3 partial observations (case 3)                                        | A new object nominated from observations that were individually discarded; "observations recovered" count |

### Checks to run after generation
- Count `AUTO` / `REVIEW` / `NOMINATE`. The measured REVIEW queue is 810 rows (section 6), driven by weak typing rather than twins, so it is accepted as a ranked queue, not tuned.
- Precision per decision band against `true_object_id` (the pipeline notebook prints `precision_vs_synthetic_truth`). Record the numbers after each run for the demo and slides, labelled **synthetic**.
- Nominations: `noms` and `recovered` (the notebook prints these). Report the recovered-observation count and its share of all observations.
- Confirm no `true_object_id` appears in the model features or the dossier prompt.

## 5. Open items
- Measured: see section 6 (band counts, precision per band, held-out AUC / average precision, nominations, hero IDs). Working `ai_query` endpoint: `databricks-meta-llama-3-3-70b-instruct`.
- CCO `Event` class choice: not researched; out of scope for the build.
- Whether a tasking feature (was this area tasked at the time?) improves the model: out of scope. The tasking table is used for lineage, governance and the "why no observations here?" question only.

## 6. Measured results

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
