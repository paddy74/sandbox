"""Analyst review screen (Layer 4): ranked review queue, candidate map, decision write-back.

One screen: pick a pending review item, see why the model scored it (match probability,
distance, type score, runner-up), see the candidate and runner-up on a map, read the dossier
if one exists, then APPROVE or REJECT. The decision is written to ``review_decisions`` and an
APPROVE also updates the object in ``oms_objects``, the same as ``adjudicate()`` in
``notebooks/generate_data.py``. All data is synthetic.

Deploy (Databricks Apps, Streamlit):
1. Create an app from this folder. Add a **SQL warehouse** resource with key ``sql-warehouse``
   (permission: Can use). ``app.yaml`` maps it to ``DATABRICKS_WAREHOUSE_ID``.
2. Grant the app's service principal access to the demo schema (run as a catalog owner)::

       GRANT USE CATALOG ON CATALOG workspace TO `<app service principal>`;
       GRANT USE SCHEMA, SELECT, MODIFY ON SCHEMA workspace.obj_resolution_demo TO `<app service principal>`;

3. Run ``notebooks/generate_data.py`` first so the tables and views exist.
"""

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


def record_decision(obs_id: str, object_id: str, decision: str) -> str:
    """Write one analyst decision; APPROVE also updates the object. Returns a status message."""
    if decision not in ("APPROVE", "REJECT"):
        raise ValueError(f"decision must be APPROVE or REJECT, got {decision!r}")
    if len(
        run(
            f"SELECT 1 FROM {SCHEMA}.review_decisions WHERE obs_id = :obs",
            {"obs": obs_id},
        )
    ):
        return f"{obs_id} was already decided; no change."
    run(
        f"INSERT INTO {SCHEMA}.review_decisions "
        "VALUES (:obs, :obj, :dec, :who, current_timestamp())",
        {"obs": obs_id, "obj": object_id, "dec": decision, "who": analyst_name()},
        fetch=False,
    )
    if decision == "APPROVE":
        run(
            f"UPDATE {SCHEMA}.oms_objects SET obs_count = obs_count + 1, "
            "last_seen = current_timestamp() WHERE object_id = :obj",
            {"obj": object_id},
            fetch=False,
        )
    return f"{decision}: {obs_id} -> {object_id}"


st.set_page_config(page_title="Object resolution analyst review", layout="wide")
st.title("Analyst review queue")
st.caption("All data is synthetic. Ranked by model match probability.")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

queue = run(f"""
    SELECT obs_id, candidate_object_id, candidate_type, match_prob, runner_up_object_id,
           runner_up_prob, dist_m, type_score, reported_type, producer, sensor, confidence,
           lat, lon
    FROM {SCHEMA}.review_queue
    WHERE analyst_decision IS NULL
    ORDER BY match_prob DESC
    LIMIT 50""")
decided = run(f"SELECT COUNT(*) AS n FROM {SCHEMA}.review_decisions").iloc[0, 0]

left, right = st.columns(2)
left.metric("Pending review (top 50 shown)", len(queue))
right.metric("Decisions recorded", int(decided))

if queue.empty:
    st.info("No pending review items.")
    st.stop()

st.dataframe(
    queue[
        [
            "obs_id",
            "candidate_object_id",
            "candidate_type",
            "match_prob",
            "runner_up_prob",
            "dist_m",
            "producer",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

labels = [
    f"{r.obs_id} | {r.candidate_object_id} | p={r.match_prob:.3f}"
    for r in queue.itertuples()
]
choice = st.selectbox("Review item", labels)
row = queue.iloc[labels.index(choice)]

info, map_col = st.columns([1, 1])
with info:
    st.subheader("Why this score")
    gap = row.match_prob - row.runner_up_prob if pd.notna(row.runner_up_prob) else None
    st.table(
        pd.DataFrame(
            {
                "value": {
                    "Match probability": f"{row.match_prob:.3f}",
                    "Runner-up": (
                        f"{row.runner_up_object_id} ({row.runner_up_prob:.3f})"
                        if pd.notna(row.runner_up_prob)
                        else "none"
                    ),
                    "Gap to runner-up": f"{gap:.3f}" if gap is not None else "n/a",
                    "Distance to candidate (m)": f"{row.dist_m:.0f}",
                    "Type score": f"{row.type_score:.1f}",
                    "Reported type": row.reported_type or "missing",
                    "Candidate type": row.candidate_type,
                    "Producer / sensor": f"{row.producer} / {row.sensor}",
                    "Confidence": f"{row.confidence:.2f}",
                }
            }
        )
    )

with map_col:
    st.subheader("Where")
    ids = [row.candidate_object_id, row.runner_up_object_id or row.candidate_object_id]
    objs = run(
        f"SELECT object_id, object_type, lat, lon FROM {SCHEMA}.oms_objects "
        "WHERE object_id IN (:a, :b)",
        {"a": ids[0], "b": ids[1]},
    ).set_index("object_id")
    m = folium.Map(location=[row.lat, row.lon], zoom_start=16)
    folium.CircleMarker(
        [row.lat, row.lon], radius=8, color="#FF3621", fill=True, tooltip="observation"
    ).add_to(m)
    for oid, role in (
        (row.candidate_object_id, "candidate"),
        (row.runner_up_object_id, "runner-up"),
    ):
        if oid and oid in objs.index:
            o = objs.loc[oid]
            folium.Marker(
                [o.lat, o.lon],
                tooltip=f"{role}: {oid} ({o.object_type})",
                icon=folium.Icon(color="black" if role == "candidate" else "gray"),
            ).add_to(m)
    st_folium(m, height=360, use_container_width=True, returned_objects=[])
    st.caption(
        "Red dot: the observation. Black pin: candidate object. Grey pin: runner-up."
    )

dossier = run(
    f"SELECT dossier_text, generated_by FROM {SCHEMA}.object_dossiers WHERE object_id = :obj",
    {"obj": row.candidate_object_id},
)
st.subheader("Dossier")
if dossier.empty:
    st.info("No dossier has been generated for this candidate object.")
else:
    st.write(dossier.iloc[0].dossier_text)
    st.caption(f"Generated by: {dossier.iloc[0].generated_by}")

st.subheader("Decision")
approve, reject = st.columns(2)
if approve.button("Approve", type="primary", use_container_width=True):
    st.session_state["flash"] = record_decision(
        row.obs_id, row.candidate_object_id, "APPROVE"
    )
    st.rerun()
if reject.button("Reject", use_container_width=True):
    st.session_state["flash"] = record_decision(
        row.obs_id, row.candidate_object_id, "REJECT"
    )
    st.rerun()
st.caption(
    "Decisions are stored in review_decisions and become training labels for the next retrain."
)
