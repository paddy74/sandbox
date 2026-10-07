# 06: Slide outline

Purpose: a five-slide deck (problem, architecture, what we built, value and ROI, 30-day plan) with headlines, sparse on-slide content, visuals, speaker notes and the objection each slide pre-empts. Facts come from files 02 and 03.

Rules for every slide: synthetic data only, and the word **synthetic** appears on any slide with a demo number. No invented savings figures. Anything not verified on the workspace (Genie, Databricks Apps, model registration, lineage graph) is described as planned or "to verify", never as done. Slide time total: about 6 to 7 minutes out of the 30-minute demo-and-storytelling block, leaving the rest for the live demo (file 05).

## Slide-to-audience map

| Slide                          | Primary audience               | Pre-empts                                            |
| ------------------------------ | ------------------------------ | ---------------------------------------------------- |
| 1 Problem and cost of inaction | Business Leader                | "Why now? Is this really costing us?"                |
| 2 Architecture                 | CDO, VP Engineering            | "Another silo?" "Too complex?" "Where is the trust?" |
| 3 What we built                | All three                      | "Is it real or a slide?" "Do I trust the AI?"        |
| 4 Value and ROI                | Business Leader, CDO           | "What does it cost versus what do I get?"            |
| 5 30-day plan                  | All three, VP Engineering most | "What do you need from me, and what could go wrong?" |

---

## Slide 1: Problem and cost of inaction

**Headline:** Every brief starts with an argument about how many objects there really are.

**On-slide content (sparse)**
- Three source types, no shared view: object system (Postgres), tasking (MySQL), observation feeds (SaaS/API), plus years of report text.
- Analysts correlate overlapping observations by hand, in spreadsheets.
- Observations too incomplete to stand alone are discarded.
- Cost of waiting, as questions rather than numbers: `[hours per disputed brief - customer to supply]`, `[observations discarded per week - customer to supply]`.

**Visual suggestion:** one image of the same coordinates with three overlapping dots from three feeds and a question mark ("one object or three?"). Left: today (three sources, spreadsheet). Right: blank, filled in on slide 3. Keep it to one picture and about 30 words.

**Speaker notes (about 75 s)**
"In the discovery we heard [reflect the leader's own number from the opener: hours or days on the last disputed brief]. That is the cost to name: every brief starts by deciding how many objects are really there. The data sits in three kinds of systems that do not talk, so analysts line up observations by hand against the object records. Two observations 200 metres apart might be one object or two. When an observation is too incomplete to support a judgment alone, it is discarded, so the organization pays to collect it and then does not use it. And inaccurate or incomplete object data has visible real-world consequences, so the stakes are about trust in the brief, not tidiness. Nothing here learns: yesterday's analyst decision does not make tomorrow's match faster. The next four slides show how that changes."

**Pre-empts:** "Why now, and is this a real cost or a nuisance?" It anchors the cost in the leader's own words and number, not mine.

**Fill in before presenting:** replace the bracketed placeholders only with what the leader said; if nothing was said, leave the question form.

---

## Slide 2: Architecture across the four layers

**Headline:** One governed foundation, one model, one dossier, one screen: each layer feeds the next.

**On-slide content (sparse)**

| Layer           | One line                                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1 Governed data | Four source types land on one entity in Unity Catalog with lineage columns, ontology-aligned types and row filters          |
| 2 Predictive ML | Match probability for every observation-to-object pair: >= 0.9 auto-associate, 0.5 to 0.9 review, < 0.5 nominate new object |
| 3 GenAI         | A sourced dossier per object that cites its evidence; plain-English questions through Genie (to verify)                     |
| 4 Presentation  | Map, select object, probability and why, dossier, approve or reject; write-back to the object system                        |

**Visual suggestion:** the file-02 flow redrawn as four horizontal bands (left to right: sources, bronze, pairs, decisions, review/nominate, dossier, screen). Use one colour per layer; mark the mocked sources with a small "synthetic" tag and the unverified items (model registration, Genie, App) with a dashed outline. One arrow loops from "analyst decision" back to "training labels" and is labelled "learns from outcomes".

**Speaker notes (about 80 s)**
"Left to right. Layer one: object system, tasking, observation feeds and report text land on one entity, the Object, in one governed catalog. Every row carries where it came from and when it was ingested, and object types map to the Common Core Ontologies so 'Truck' and 'Vehicle' mean the same thing everywhere. Layer two: a model scores each observation against nearby candidate objects. Very high confidence is associated automatically; the uncertain middle goes to an analyst; the rest are clustered into candidate new objects. Layer three: for any object, a generated dossier built only from the governed records and cited by ID. Layer four: one screen the analyst works from, and their decision writes back to the object system and becomes a training label. In this demo the Postgres and MySQL sources are mocked; the production path is federation or change-data-capture, which I have not tested on this workspace. I chose boring, managed components on purpose so your team can own them."

**Pre-empts:** "Another silo?" (answer: same entity and catalog across layers), "too complex?" (answer: five features, one model, one SQL clustering step), "where is the trust?" (answer: lineage columns, analyst stays decider). Trade-offs to have ready are in file 02, section 3.

**Honesty flags on this slide:** MLflow logging, Unity Catalog registration, Genie, Apps and the lineage graph are checked by the feasibility notebook (section 5 and the manual checks in section 12). If any is not verified in the target workspace when presenting, label it "planned" on the slide.

---

## Slide 3: What we built

**Headline:** In [fill: actual build time] of live building, three synthetic cases show the whole loop.

(Edit the headline to the true elapsed build time. Do not say an hour if it was not.)

**On-slide content (sparse)**

| Hero                        | What it shows                                                                                                                                    | Synthetic result                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| A: Armored Fighting Vehicle | Algorithm and analyst agree, auto-associated at >= 0.9                                                                                           | match_prob 0.985 (mean of 5 observations, `OBJ-00599`)               |
| B: look-alike Trucks        | A Truck observation sits near two look-alike records; the top candidate is 0.854, below 0.9, so an analyst decides and the object system updates | match_prob 0.854 vs runner-up 0.304 (`OBJ-T005`)                     |
| C: nominated object         | Four observations, each generic or missing type, become one new object                                                                           | 146 nominations, 352 observations recovered (7.9%); `NOM-406859eed6` |

Footer: "All data synthetic. Precision AUTO 98.3%, REVIEW 94.2%, NOMINATE 90.5% (in-sample, model) measured against synthetic ground truth; says nothing about real-world accuracy."

**Visual suggestion:** three screenshots from the actual demo (map with Hero A selected, review queue with Hero B, nomination with Hero C), each with a one-line caption. Take the screenshots during the rehearsal run so they exist if the live demo fails. Add a small funnel: `[total observations]` to `[AUTO]` / `[REVIEW]` / `[NOMINATE]` counts, all marked synthetic.

**Speaker notes (about 70 s)**
"Three cases, chosen on purpose. A: the easy win. Two sources agree, the model is above 0.9, no analyst time spent. B: the hard case. Two records 200 metres apart and a generic 'vehicle' report that fits both. That is the 'how many objects are there' argument, and here it has a probability, a reason, and one analyst decision that moves the count and writes back. C: the discarded data. Three observations, each unusable alone: a missing type, a generic type, a low-confidence one. Together they nominate a new object. These numbers are from synthetic data I generated, with ground truth I planted, so the precision I show says nothing about your real-world accuracy; the real accuracy is something we measure together in the first 30 days. I will show all of this live next." Then go to the live demo.

**Pre-empts:** "Is this real or just slides?" and "Do I trust the AI?" Reliability controls to mention if asked: the dossier is restricted to the retrieved rows and cites IDs, the demo checks cited IDs exist, the dossier never changes object data, the analyst approves (file 02, Layer 3).

**Honesty flags:** `true_object_id` stands in for past adjudications and is never a model feature; say so. Do not claim a retrain ran unless it did.

---

## Slide 4: Value and ROI

**Headline:** The value is analyst time, trusted counts and recovered observations; we validate the numbers with you before any commitment.

**On-slide content (sparse)**

Formula (left):

```
Analyst hours saved / month
  = correlations per month            [V   - customer to supply]
  x analyst minutes per correlation   [M   - customer to supply]
  x share auto-associated             [S   - measure in pilot; demo value synthetic]
  / 60

Observations recovered / month
  = observations per month            [O   - customer to supply]
  x share currently discarded         [D   - customer to supply]
  x share recoverable via clustering  [R   - measure in pilot; demo value synthetic]

Time to brief
  = today's elapsed time on a disputed count    [T0 - customer to supply]
  vs pilot elapsed time                         [T1 - measure in pilot]
```

Assumptions table (right): each row is an input, who supplies it, and how it is validated.

| Input                                | Source                                       | Validated by                                    |
| ------------------------------------ | -------------------------------------------- | ----------------------------------------------- |
| Correlations per month, minutes each | Analyst leads                                | Two-week time sample                            |
| Share currently discarded            | Collection managers                          | Count of observations never associated          |
| Share auto-associated (S)            | Pilot, on the customer's adjudicated history | Precision at the 0.9 threshold vs adjudications |
| Cost of a wrong merge or split       | Business Leader                              | Sets the thresholds, not a savings number       |

Demo values, labelled: "Synthetic demo: S = 66.6% auto-associated (in-sample, 4,474 observations), 18.1% routed to review, recovered = 352 observations (7.9%), synthetic". **Do not put any dollar figure on this slide.**

**Visual suggestion:** a two-column slide, formula on the left in large type with blanks, assumptions table on the right. Optionally a before/after bar for "time to settle a disputed count" with the left bar labelled "today: [customer to supply]" and the right "pilot: [measure]". No invented bar heights; use blank placeholders if no data.

**Speaker notes (about 90 s)**
"I am not going to give you a savings number today, because it would be mine, not yours. Here is how we build one together. Analyst hours saved are the number of correlations, times the minutes each takes, times the share the system handles safely. You told me [reflect the number from discovery]. The share the system handles safely is something we measure on your own adjudication history in the pilot, not something I assert; the synthetic demo value is [S, labelled synthetic]. Second: recovered observations. These are the ones you collect and discard today; the demo's synthetic recovery rate is [R, synthetic], and yours is a pilot measurement. Third: time to brief. Hours versus weeks is the goal you described; we measure elapsed time on a disputed count before and after. The other side of ROI is cost: the pilot uses managed serverless components, so there is no new infrastructure to staff; actual licensing and compute cost I will scope with your account team, not guess here. Thresholds are a business decision, set by the cost of a wrong merge versus a missed one, and your analysts own that dial."

**Pre-empts:** cost ("what do I pay and what do I get"), trust in the data (the thresholds and the review queue keep the analyst in control), and AI reliability (precision is measured on your history before anything is automated). It also pre-empts the "show me ROI" trap by converting it into validated inputs with owners.

**Honesty flags:** no cost figures for Databricks pricing on this slide. Do not cite any industry savings statistic unless verified with a public source.

---

## Slide 5: 30-day plan and expansion

**Headline:** In 30 days we measure it on your data, in your environment, with your analysts deciding.

**On-slide content (sparse)**

| Week | Milestone                                                                                                                                               | Owner                                                                   | Success criterion                                                                                                                                   |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Environment and authorization path confirmed; data access path agreed; named champion and 3 to 5 analysts engaged; baseline time sample started         | Customer (environment, champion); delivery team (plan, baseline design) | Written decision on where this runs; read access to a sample of objects and observations; baseline method agreed                                    |
| 2    | Governed ingestion of one object class and two sources into the customer's catalog; type mapping to the customer's ontology; first correlation baseline | Delivery team with customer data engineers                              | Row counts reconcile to source; lineage and ingestion audit columns present; data quality checks defined                                            |
| 3    | Match model trained on the customer's adjudication history; thresholds proposed; review queue live for a small analyst group                            | Delivery team with analyst lead                                         | Precision per probability band measured on held-out adjudications, reported to the leader, with the false-merge versus false-split trade-off agreed |
| 4    | Dossier over governed reports; side-by-side comparison with the current process; readout to the leader, CDO and engineering                             | Delivery team, customer champion                                        | Agreed ROI inputs measured (slide 4); go / no-go on expansion with all three audiences present                                                      |

**What we need from you (bold on the slide):** (1) a real data access path, (2) the authorization environment decision, (3) analyst adjudication history for labels, (4) a named champion.

**Risks, one line each:** authorization environment not available in time; adjudication history thin or inconsistent; object type mapping takes longer than expected; analyst adoption.

**Expansion (one line):** more object classes, events, more feeds, retraining from the review queue, and an application for the analyst workflow once the pilot proves out.

**Visual suggestion:** a four-column timeline (week 1 to week 4) with a lock icon on week 1 labelled "gating: environment and authorization" and an arrow from "analyst decisions" back to "model" labelled "retrain". A small box at the right: "What we need from you".

**Speaker notes (about 90 s)**
"If you say go, the first week is not about building. It is about two decisions only you can make: where this runs, and who owns it. The authorization environment is a gating item. I am not going to claim a specific authorization level for the platform today; I will confirm exactly what applies to your environment with our public sector team before we plan anything, and that confirmation is a week-one deliverable. I will also need a path to read-only data, a champion on your side, and the history of analyst adjudications, because that history is what trains and validates the model; without it the model starts as a ranker and learns as analysts review. Week two we ingest one object class from two sources with lineage. Week three we train and measure precision by probability band on your held-out decisions and agree the thresholds together. Week four we compare side by side with how you work today and read out with the leader, the data office and engineering in the room. Risks: environment timing, thin label history, and ontology mapping, which I would rather tell you now than discover in week three. Success in 30 days is a measured number that you believe, not a finished product."

**Pre-empts:** security and ATO ("will this get through authorization?" answered honestly as a week-one gating item), complexity ("what exactly is the scope?"), and the "pilot that never ends" fear (explicit week-4 go / no-go).

**Honesty flags (must be true when said):**
- Do not assert any FedRAMP, IL4/IL5 or other authorization level for Databricks. Verify with an authoritative public source or the public sector team before presenting, and cite only if verified. If not verified: "I will confirm and come back to you in writing."
- Week plan assumes read access to real data is possible; if only synthetic or exported data is allowed in the first two weeks, say the pilot proceeds on exports and the ingestion path is proven later.
- Federation to the real Postgres and MySQL was not tested on this workspace; the week-2 ingestion path may be federation, CDC, or exports depending on the customer environment.

---

## Deck-wide checklist before presenting

| Check                                                                                       | Done |
| ------------------------------------------------------------------------------------------- | ---- |
| Every placeholder in brackets is filled from the latest run or removed                      | [ ]  |
| Every slide with a demo number says synthetic                                               | [ ]  |
| No dollar savings figures anywhere                                                          | [ ]  |
| Slide 2 and 3 mark anything unverified (registration, Genie, App, lineage graph) as planned | [ ]  |
| Slide 3 screenshots taken in rehearsal (backup if the live demo fails)                      | [ ]  |
| Authorization claim removed or verified with a public source                                | [ ]  |
| Hero object IDs on a one-line cheat card                                                    | [ ]  |
| Slide count is 5 or fewer                                                                   | [ ]  |
| Each slide has a one-sentence takeaway the presenter can say without notes                  | [ ]  |
