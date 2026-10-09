# Object Resolution Demo

Observation-to-object resolution on Databricks. This repository is the build workspace for the demo. All data is **synthetic**.

## Overview

Analysts manually correlate overlapping observations from algorithms and single-source analysts against existing objects. Duplicate and conflicting records inflate counts, consume analyst hours and erode trust, and incomplete observations are discarded even when, correlated with others, they could support a judgment.

The demo resolves incoming observations into one trusted, explainable record per object across four layers:

- **Governed data foundation**: mocked object, collection and observation sources unified on the Object, aligned to the public BFO and CCO ontologies, governed in Unity Catalog.
- **Predictive ML**: a match-probability model that auto-associates, routes to analyst review or nominates new objects, retrained from analyst decisions.
- **GenAI**: sourced object dossiers and plain-English questions over the governed data.
- **Presentation**: an analyst app with a map, dossiers and a review queue that writes back to the system of record.

## Layout

| Path              | Purpose                                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `obj_resolution/` | Shared helpers used by notebooks and the app                                                                                                            |
| `notebooks/`      | Feasibility checks and pipeline notebooks (Databricks `.py` source format)                                                                              |
| `app/`            | Databricks App; its dependencies are in `app/requirements.txt`                                                                                          |
| `data/ontology/`  | `CommonCoreOntologiesMerged.ttl` from the [Common Core Ontologies](https://github.com/CommonCoreOntology/CommonCoreOntologies) repo (`src/cco-merged/`) |
| `docs/plan/`      | Planning documents ([index](docs/plan/README.md))                                                                                                       |
| `demo/`           | Genie example queries and space instructions                                                                                                            |
| `tests/unit/`     | Tests for `obj_resolution/` helpers                                                                                                                     |

## Setup on Databricks

1. Clone this repo as a **git folder** in the workspace.
2. Attach notebooks to **serverless** compute.
3. Install the repo's dependencies in the first cell of each notebook, then restart Python:

   ```python
   %pip install -e /Workspace/Users/<you>/<repo-folder>
   ```

   ```python
   dbutils.library.restartPython()
   ```

   Alternatively, add the same path under the notebook's **Environment** panel dependencies.
4. Run `notebooks/feasibility_tests.py` top to bottom in Databricks before planning or building a design that depends on a platform feature or package; review the summary table and adopt the fallback for anything that fails.

## Deploy order

Nothing is deployed by default; each step is done in the target workspace.

1. Run `notebooks/feasibility_tests.py` and complete its manual checks (section 12).
2. Run `notebooks/generate_data.py` top to bottom to create the schema, tables, model and views (stage code is in `notebooks/pipeline/`).
3. Create the Databricks App from `app/` (SQL warehouse resource and schema grants are in the `app/app.py` docstring).
4. The Genie space is created from `demo/genie-examples.md` by step 2 (section 15); hide `true_object_id` and `dup_of` in its UI once. Optionally add an AI/BI dashboard (the fallback to the app).

## Local development (optional)

```bash
uv sync --all-groups
uv run pytest
uv run ruff check
uv run ruff format
```

## Design

The design prioritizes **reliability and explainability over sophistication**, so every step can be demoed live and explained to a non-technical audience.

- **`COPY INTO` ingestion**: incremental and idempotent without a streaming query, so re-runs are safe and fast.
- **Ontology-aware matching**: CCO class hierarchy drives type compatibility, so generic and specific reports can be correlated.
- **Probabilities, not rules**: a trained model with thresholds and a human review queue, so reliability questions have a measurable answer.
- **Write-back**: resolved objects return to the system of record, so the platform enriches existing systems instead of becoming another silo.
