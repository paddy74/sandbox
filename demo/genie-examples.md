# Genie examples: curated prompts and SQL

This file is the source of the Genie space: section 15 of `notebooks/generate_data.py` (`sync_space` in `notebooks/pipeline/genie.py`) creates or updates the space from it (the instructions block, one example SQL per `###` heading, sample questions 1, 5 and 2, and every table the examples query). Edit here and re-run that cell; a section without a `sql` block fails the cell. All tables are in `workspace.obj_resolution_demo` (the cell swaps in the current catalog). All data is synthetic. Hide `true_object_id` and `dup_of` in the Genie UI once after the space is created; updates keep that, and none of these queries use them.

I wrote these against the notebook's schema and could not run them on your workspace, so **run each once in the SQL editor** before saving it as a trusted example.

## Space instructions

```markdown
Data is synthetic. An observation is one report of something seen. An object is a tracked thing
(Airport, Military Facility, Truck, Armored Fighting Vehicle, Aircraft) in the object system.
- oms_objects: authoritative objects. source = 'OMS' (original) or 'DATABRICKS_NOMINATION' (new, status 'NOMINATED').
  Name an object as object_type + ' ' + designator (e.g. 'Truck Bravo-12'); object_id is a system key.
- bronze_observations: raw observations (obs_time, reported_type, producer, sensor, confidence, source_file).
- silver_model_decisions: one row per observation. decision is AUTO (match_prob >= 0.9, associated to matched_object_id),
  REVIEW (0.5 to 0.9, waits for an analyst) or NOMINATE (< 0.5, clustered into a new object).
- review_queue: the REVIEW observations ranked by match_prob; analyst_decision is NULL until an analyst decides.
- gold_nominations: new objects created from clusters of unmatched observations.
- object_dossiers: sourced summaries; cited_ids are report IDs.
Count "unique objects with observations" through silver_model_decisions.matched_object_id where decision = 'AUTO'.
Reported types Facility, Ground Vehicle and Vehicle are generic (weak); NULL is missing.
```

## Examples

### 1. How many observations fall in each decision band?
```sql
SELECT decision,
       COUNT(*) AS observations,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM workspace.obj_resolution_demo.silver_model_decisions
GROUP BY decision
ORDER BY decision;
```

### 2. How many observations are awaiting analyst review?
```sql
SELECT COUNT(*) AS awaiting_review
FROM workspace.obj_resolution_demo.review_queue
WHERE analyst_decision IS NULL;
```

### 3. Show the top 10 review items by match probability
```sql
SELECT obs_id, candidate_object_id, candidate_type, match_prob, runner_up_object_id, runner_up_prob,
       dist_m, producer, sensor, confidence
FROM workspace.obj_resolution_demo.review_queue
WHERE analyst_decision IS NULL
ORDER BY match_prob DESC
LIMIT 10;
```

### 4. Which review items are ambiguous between two objects?
```sql
SELECT obs_id, candidate_object_id, match_prob, runner_up_object_id, runner_up_prob,
       ROUND(match_prob - runner_up_prob, 3) AS gap
FROM workspace.obj_resolution_demo.review_queue
WHERE runner_up_prob IS NOT NULL
  AND match_prob - runner_up_prob <= 0.2
ORDER BY gap ASC;
```

### 5. How many unique Armored Fighting Vehicle objects were observed in the last 72 hours?
Counts objects with at least one AUTO-associated observation. REVIEW observations are still pending, so they are not counted.
```sql
SELECT COUNT(DISTINCT o.object_id) AS unique_objects
FROM workspace.obj_resolution_demo.silver_model_decisions d
JOIN workspace.obj_resolution_demo.bronze_observations b ON b.obs_id = d.obs_id
JOIN workspace.obj_resolution_demo.oms_objects o ON o.object_id = d.matched_object_id
WHERE d.decision = 'AUTO'
  AND o.object_type = 'Armored Fighting Vehicle'
  AND b.obs_time >= current_timestamp() - INTERVAL 72 HOURS;
```
Note: observation times are stamped relative to the moment the notebook ran. If you present hours later, the 72-hour window drops the oldest observations, so the count falls slightly. Re-run the notebook shortly before the demo for a clean number.

### 6. How many objects are there by type and status?
```sql
SELECT object_type, status, COUNT(*) AS objects
FROM workspace.obj_resolution_demo.oms_objects
GROUP BY object_type, status
ORDER BY object_type, status;
```

### 7. How many new objects were nominated, and how many observations did that recover?
```sql
SELECT COUNT(*) AS nominations,
       SUM(obs_count) AS observations_recovered
FROM workspace.obj_resolution_demo.gold_nominations;
```

### 8. Which nominated objects have the most supporting observations?
```sql
SELECT concat(object_type, ' ', designator) AS object_name, obs_count, lat, lon
FROM workspace.obj_resolution_demo.gold_nominations
ORDER BY obs_count DESC, object_id
LIMIT 10;
```

### 9. What share of observations is auto-associated, by producer?
```sql
SELECT b.producer,
       COUNT(*) AS observations,
       SUM(CASE WHEN d.decision = 'AUTO' THEN 1 ELSE 0 END) AS auto_associated,
       ROUND(100.0 * SUM(CASE WHEN d.decision = 'AUTO' THEN 1 ELSE 0 END) / COUNT(*), 1) AS auto_pct
FROM workspace.obj_resolution_demo.silver_model_decisions d
JOIN workspace.obj_resolution_demo.bronze_observations b ON b.obs_id = d.obs_id
GROUP BY b.producer
ORDER BY b.producer;
```

### 10. Why is the review queue so large? Show decisions by how specific the reported type is
This is the answer to "why 810 rows": weakly typed observations cannot clear the 0.9 threshold.
```sql
SELECT CASE WHEN b.reported_type IS NULL THEN 'missing type'
            WHEN b.reported_type IN ('Facility', 'Ground Vehicle', 'Vehicle') THEN 'generic type'
            ELSE 'specific type' END AS type_quality,
       d.decision,
       COUNT(*) AS observations
FROM workspace.obj_resolution_demo.silver_model_decisions d
JOIN workspace.obj_resolution_demo.bronze_observations b ON b.obs_id = d.obs_id
GROUP BY 1, 2
ORDER BY 1, 2;
```

### 11. Show the dossier for object OBJ-00599
Replace the ID with hero A or B from the cheat card (A: `OBJ-00599`, B: `OBJ-T005`).
```sql
SELECT object_id, hero, dossier_text, cited_ids, generated_by
FROM workspace.obj_resolution_demo.object_dossiers
WHERE object_id = 'OBJ-00599';
```

### 12. Which files did the observations come from, and when did they land?
Supports the "every row traces to a source" claim.
```sql
SELECT source_file,
       COUNT(*) AS rows_landed,
       MIN(ingested_at) AS first_ingested,
       MAX(ingested_at) AS last_ingested
FROM workspace.obj_resolution_demo.bronze_observations
GROUP BY source_file
ORDER BY source_file;
```

### 13. Which sensors feed the review queue?
```sql
SELECT sensor, COUNT(*) AS review_items, ROUND(AVG(match_prob), 3) AS avg_match_prob
FROM workspace.obj_resolution_demo.review_queue
GROUP BY sensor
ORDER BY review_items DESC;
```

### 14. Which objects have the most linked observations?
```sql
SELECT concat(object_type, ' ', designator) AS object_name, source, status, obs_count
FROM workspace.obj_resolution_demo.oms_objects
ORDER BY obs_count DESC, object_id
LIMIT 10;
```

### 15. How many analyst decisions have been recorded?
`review_decisions` is empty until hero B's approval cell has run.
```sql
SELECT decision, COUNT(*) AS decisions
FROM workspace.obj_resolution_demo.review_decisions
GROUP BY decision;
```

## Which to use live
- **Core demo (3):** 1, 5 (the plain-English question from the brief), and 2.
- **Trust and lineage:** 12.
- **If asked "why is the queue big":** 10.
- **To show the dossier:** 11, after the notebook has built it.

## Phrasings to test as natural-language variants
"How many Armored Fighting Vehicles were seen in the last three days", "Which observations need review", "How many new objects did we find", "Show me the dossier for the clean Armored Fighting Vehicle". If Genie gets one wrong, add that variant as another example rather than editing these.
