"""Analyst review screen (Layer 4): ranked review queue, candidate map, decision write-back.

One screen: pick a pending observation, see in plain language why the model suggested an
object (how sure it is, how far away, whether the type agrees, the next-best object), see both
on a map, read the source report and the dossier if one exists, then confirm or reject the
link with a required ICD 203 confidence (High, Moderate, Low). Scores appear only as ICD 203
likelihood terms. The decision is written to ``review_decisions`` and a confirm also updates
the object in ``oms_objects``, the same as ``adjudicate()`` in ``notebooks/pipeline/review.py``.
System IDs are kept out of the main view and shown only as references. All data is synthetic.

Deploy (Databricks Apps, Streamlit):
1. Create an app from this folder. Add a **SQL warehouse** resource with key ``sql-warehouse``
   (permission: Can use). ``app.yaml`` maps it to ``DATABRICKS_WAREHOUSE_ID``.
2. Grant the app's service principal access to the demo schema (run as a catalog owner)::

       GRANT USE CATALOG ON CATALOG workspace TO `<app service principal>`;
       GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA workspace.obj_resolution_demo TO `<app service principal>`;

3. Run ``notebooks/generate_data.py`` first so the tables and views exist.
"""

import math
import os
import re

import folium
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config
from streamlit_folium import st_folium

SCHEMA = os.getenv("OBJ_RESOLUTION_SCHEMA", "workspace.obj_resolution_demo")
if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", SCHEMA):
    raise ValueError(
        f"OBJ_RESOLUTION_SCHEMA must look like catalog.schema, got {SCHEMA!r}"
    )
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
if not WAREHOUSE_ID:
    raise RuntimeError(
        "DATABRICKS_WAREHOUSE_ID is not set: add a SQL warehouse resource with key "
        "'sql-warehouse' to the app (see app.yaml)"
    )

AUTO_T = 0.9  # auto-link threshold, Config.auto_t in notebooks/pipeline/config.py
MAX_DIST_M = 500  # candidate search radius in notebooks/generate_data.py
CLOSE_CALL_GAP = 0.15  # runner-up within this many points: say it is a close call

SENSORS = {
    "EO": "Electro-optical imagery",
    "SAR": "Radar imagery (SAR)",
    "FMV": "Full-motion video",
}
PRODUCERS = {"algorithm": "Automated detection", "analyst": "Analyst report"}
# ICD 203 analytic confidence, as in notebooks/pipeline/icd203.py. Scores are shown only as
# ICD 203 likelihood terms, which the review_queue view maps from the numbers.
CONFIDENCE = ["High", "Moderate", "Low"]

cfg = Config()


def run(query: str, params: dict | None = None, fetch: bool = True) -> pd.DataFrame:
    """Run one statement on the SQL warehouse; return rows as a DataFrame (empty if not fetching)."""
    with (
        sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
            credentials_provider=lambda: cfg.authenticate,
        ) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(query, params or {})
        return cur.fetchall_arrow().to_pandas() if fetch else pd.DataFrame()


def analyst_name() -> str:
    """Signed-in user from the Databricks Apps proxy header, else a demo label."""
    try:
        return st.context.headers.get("X-Forwarded-Email") or "analyst_demo"
    except Exception:
        return "analyst_demo"


def coords(lat: float, lon: float) -> str:
    """Readable coordinates, e.g. 39.2134° N, 105.3312° W."""
    return f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.4f}° {'E' if lon >= 0 else 'W'}"


def object_label(object_type: str, designator: str | None) -> str:
    """Human-readable object name: CCO type plus designator, e.g. 'Truck Bravo-12'."""
    if not designator or pd.isna(designator):
        raise ValueError(
            f"{object_type} object has no designator: re-run notebooks/generate_data.py"
        )
    return f"{object_type} {designator}"


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def time_ago(ts: pd.Timestamp, now: pd.Timestamp) -> str:
    """Age of a timestamp in words, e.g. '5 h ago' or '2 days ago'; both times are UTC."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:  # the SQL connector may return tz-aware UTC
        ts = ts.tz_convert(None)
    hours = (now - ts).total_seconds() / 3600
    if hours < 1:
        return "under 1 h ago"
    if hours < 48:
        return f"{hours:.0f} h ago"
    return f"{hours / 24:.0f} days ago"


def source_label(producer: str, sensor: str) -> str:
    """Who produced an observation and from what, e.g. 'Analyst report, Full-motion video'."""
    return f"{PRODUCERS.get(producer, producer)}, {SENSORS.get(sensor, sensor)}"


def type_agreement(reported: str | None, candidate: str) -> str:
    """Plain-language version of the model's type score."""
    if reported is None:
        return "Source did not say what it saw (neutral)"
    if reported == candidate:
        return f"Agrees: reported as {reported}"
    return f"Compatible: reported only as a general '{reported}'"


def record_decision(
    obs_id: str, object_id: str, decision: str, confidence: str, label: str
) -> str:
    """Write one analyst decision with its ICD 203 confidence; APPROVE also updates the object.

    :return: a status message for the analyst.
    """
    if decision not in ("APPROVE", "REJECT"):
        raise ValueError(f"decision must be APPROVE or REJECT, got {decision!r}")
    if confidence not in CONFIDENCE:
        raise ValueError(f"confidence must be one of {CONFIDENCE}, got {confidence!r}")
    if len(
        run(
            f"SELECT 1 FROM {SCHEMA}.review_decisions WHERE obs_id = :obs",
            {"obs": obs_id},
        )
    ):
        return "This observation was already decided; nothing changed."
    run(
        f"INSERT INTO {SCHEMA}.review_decisions "
        "(obs_id, object_id, decision, confidence, decided_by, decided_at) "
        "VALUES (:obs, :obj, :dec, :conf, :who, current_timestamp())",
        {
            "obs": obs_id,
            "obj": object_id,
            "dec": decision,
            "conf": confidence,
            "who": analyst_name(),
        },
        fetch=False,
    )
    if decision == "REJECT":
        return f"Rejected ({confidence.lower()} confidence): the observation is not linked to {label}."
    # last_seen is when the object was observed (obs_time), not when the analyst decided.
    run(
        f"MERGE INTO {SCHEMA}.oms_objects t "
        f"USING (SELECT :obj AS object_id, obs_time FROM {SCHEMA}.bronze_observations "
        "WHERE obs_id = :obs) s ON t.object_id = s.object_id "
        "WHEN MATCHED THEN UPDATE SET t.obs_count = t.obs_count + 1, "
        "t.last_seen = greatest(t.last_seen, s.obs_time)",
        {"obj": object_id, "obs": obs_id},
        fetch=False,
    )
    n = run(
        f"SELECT obs_count FROM {SCHEMA}.oms_objects WHERE object_id = :obj",
        {"obj": object_id},
    )
    total = f" It now has {int(n.iloc[0, 0])} linked observations." if len(n) else ""
    return f"Confirmed ({confidence.lower()} confidence): the observation is linked to {label}.{total}"


st.set_page_config(page_title="Object resolution review", layout="wide")
st.title("Observations awaiting analyst review")
st.caption(
    "Each row is a new sighting that the model could link to a known object but not with "
    "enough certainty to link automatically. Likelihoods use the ICD 203 terms; most likely "
    "matches first. All data is synthetic."
)
st.caption(
    f"Note: matches of {AUTO_T:.0%} or more are linked automatically, without review."
)

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

queue = run(f"""
    SELECT q.obs_id, q.obs_time, q.lat, q.lon, q.reported_type, q.producer, q.sensor,
           q.likelihood, q.match_prob, q.match_likelihood, q.dist_m, q.n_candidates, q.days_since_last_seen,
           q.candidate_object_id, q.candidate_type, c.designator AS cand_designator,
           c.lat AS cand_lat, c.lon AS cand_lon,
           c.obs_count AS cand_obs_count,
           q.runner_up_object_id, q.runner_up_prob, q.runner_up_likelihood,
           r.object_type AS runner_up_type,
           r.designator AS ru_designator,
           r.lat AS ru_lat, r.lon AS ru_lon,
           rp.report_text, h.hero
    FROM {SCHEMA}.review_queue q
    JOIN {SCHEMA}.oms_objects c ON c.object_id = q.candidate_object_id
    LEFT JOIN {SCHEMA}.oms_objects r ON r.object_id = q.runner_up_object_id
    LEFT JOIN {SCHEMA}.bronze_reports rp ON rp.obs_id = q.obs_id
    LEFT JOIN {SCHEMA}.demo_heroes h ON h.obs_id = q.obs_id
    WHERE q.analyst_decision IS NULL
    ORDER BY q.match_prob DESC
    LIMIT 50""")
counts = run(f"""
    SELECT (SELECT COUNT(*) FROM {SCHEMA}.review_queue WHERE analyst_decision IS NULL) AS pending,
           (SELECT COUNT(*) FROM {SCHEMA}.review_decisions) AS decided,
           (SELECT COUNT(*) FROM {SCHEMA}.oms_objects) AS objects""").iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("Awaiting review", f"{int(counts.pending):,}")
m2.metric("Decisions made", f"{int(counts.decided):,}")
m3.metric("Objects in the object system", f"{int(counts.objects):,}")

if queue.empty:
    st.info("Nothing awaiting review.")
    st.stop()

now = pd.Timestamp.now(tz="UTC").tz_localize(None)
queue["candidate"] = [
    object_label(r.candidate_type, r.cand_designator) for r in queue.itertuples()
]
queue["seen"] = [time_ago(t, now) for t in queue.obs_time]
queue["source"] = [source_label(r.producer, r.sensor) for r in queue.itertuples()]
queue["reported"] = queue.reported_type.fillna("Not stated")
queue["note"] = queue.hero.map(lambda h: f"Demo case {h}" if pd.notna(h) else "")

st.subheader(f"Review queue (top {len(queue)})")
st.caption("Select a row to review it.")
picked = st.dataframe(
    queue[
        [
            "match_likelihood",
            "candidate",
            "reported",
            "dist_m",
            "source",
            "seen",
            "note",
        ]
    ],
    column_config={
        "match_likelihood": "Model assessment",
        "candidate": "Suggested object",
        "reported": "Reported as",
        "dist_m": st.column_config.NumberColumn("Distance", format="%d m"),
        "source": "Source",
        "seen": "Observed",
        "note": "",
    },
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)
rows = picked.selection.rows
row = queue.iloc[rows[0] if rows else 0]

st.divider()
st.subheader(f"Should this observation be linked to {row.candidate}?")
if not rows:
    st.caption("Showing the top item; select another row above to change.")

info, map_col = st.columns([1, 1])
with info:
    has_runner_up = pd.notna(row.runner_up_prob) and pd.notna(row.runner_up_type)
    gap = row.match_prob - row.runner_up_prob if has_runner_up else None
    if gap is not None and gap < CLOSE_CALL_GAP:
        st.warning(
            f"Close call: {object_label(row.runner_up_type, row.ru_designator)} nearby is almost "
            f"as likely a match (model assessment: {row.runner_up_likelihood.lower()}, against "
            f"{row.match_likelihood.lower()} for {row.candidate}). Check the map."
        )
    else:
        st.info(
            f"The model assesses this match as {row.match_likelihood.lower()}: not "
            "certain enough to link it automatically."
        )

    st.markdown("**The observation**")
    st.table(
        pd.DataFrame(
            {
                "": {
                    "Source": source_label(row.producer, row.sensor),
                    "Reported as": type_agreement(
                        row.reported_type, row.candidate_type
                    ),
                    "Source's likelihood": row.likelihood,
                    "Observed": f"{row.seen} ({row.obs_time:%d %b %Y %H:%M} UTC)",
                    "Location": coords(row.lat, row.lon),
                }
            }
        )
    )
    if pd.notna(row.report_text):
        st.markdown("**Source report**")
        st.markdown(f"> {row.report_text}")

    st.markdown("**Why the model suggests this object**")
    runner_up = (
        f"{object_label(row.runner_up_type, row.ru_designator)}: "
        f"{row.runner_up_likelihood.lower()}, "
        f"{distance_m(row.lat, row.lon, row.ru_lat, row.ru_lon):,.0f} m away"
        if has_runner_up
        else "None: no other object is a plausible match"
    )
    st.table(
        pd.DataFrame(
            {
                "": {
                    "Model assessment": row.match_likelihood,
                    "Distance from observation": f"{row.dist_m:,.0f} m",
                    "Object location": coords(row.cand_lat, row.cand_lon),
                    "Type": type_agreement(row.reported_type, row.candidate_type),
                    "Next most likely object": runner_up,
                    "Candidate objects in range": (
                        f"{int(row.n_candidates)} within {MAX_DIST_M} m with a compatible type"
                    ),
                    "Object history": (
                        f"{int(row.cand_obs_count)} linked observations; previously seen "
                        f"{row.days_since_last_seen:.0f} days before this observation"
                    ),
                }
            }
        )
    )

with map_col:
    st.markdown("**Where**")
    m = folium.Map(location=[row.lat, row.lon], zoom_start=16)
    folium.CircleMarker(
        [row.lat, row.lon],
        radius=8,
        color="#FF3621",
        fill=True,
        tooltip=f"This observation ({row.reported})",
    ).add_to(m)
    folium.Marker(
        [row.cand_lat, row.cand_lon],
        tooltip=f"Suggested: {row.candidate} ({row.match_likelihood.lower()})",
        icon=folium.Icon(color="black"),
    ).add_to(m)
    if has_runner_up:
        folium.Marker(
            [row.ru_lat, row.ru_lon],
            tooltip=(
                f"Next most likely: {object_label(row.runner_up_type, row.ru_designator)} "
                f"({row.runner_up_likelihood.lower()})"
            ),
            icon=folium.Icon(color="gray"),
        ).add_to(m)
    st_folium(m, height=420, use_container_width=True, returned_objects=[])
    st.caption(
        "Red dot: the observation. Black pin: suggested object. "
        "Grey pin: next most likely object. Hover for details."
    )

dossier = run(
    f"SELECT dossier_text, generated_by FROM {SCHEMA}.object_dossiers WHERE object_id = :obj",
    {"obj": row.candidate_object_id},
)
with st.expander("AI summary of the suggested object", expanded=not dossier.empty):
    if dossier.empty:
        st.write("No AI summary has been generated for this object yet.")
    else:
        st.write(dossier.iloc[0].dossier_text)
        st.caption(
            f"Generated by {dossier.iloc[0].generated_by} from governed tables only. "
            "Bracketed IDs are the source reports it cites."
        )

st.subheader("Your decision")
confidence = st.selectbox(
    "Your confidence in this judgement (ICD 203)",
    CONFIDENCE,
    index=None,
    placeholder="Choose before confirming or rejecting",
    key=f"confidence_{row.obs_id}",  # a new item starts unset
    help="Confidence reflects the quality of the sources and reasoning behind your "
    "judgement, not how likely the match is.",
)
approve, reject = st.columns(2)
if approve.button(
    "Confirm: same object",
    type="primary",
    use_container_width=True,
    disabled=confidence is None,
):
    st.session_state["flash"] = record_decision(
        row.obs_id, row.candidate_object_id, "APPROVE", confidence, row.candidate
    )
    st.rerun()
if reject.button(
    "Reject: not this object", use_container_width=True, disabled=confidence is None
):
    st.session_state["flash"] = record_decision(
        row.obs_id, row.candidate_object_id, "REJECT", confidence, row.candidate
    )
    st.rerun()
st.caption(
    "Confirming links the observation to the object in the object system. "
    "Every decision is saved and used to train the next version of the model."
)

with st.expander("System references"):
    st.caption("For audit and support; not needed to make a decision.")
    st.table(
        pd.DataFrame(
            {
                "": {
                    "Observation ID": row.obs_id,
                    "Suggested object ID": row.candidate_object_id,
                    "Next most likely object ID": row.runner_up_object_id or "none",
                    "Decision table": f"{SCHEMA}.review_decisions",
                }
            }
        )
    )
