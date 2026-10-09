"""Governance demo, hero map and the views behind the AI/BI dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import folium
    from pyspark.sql import SparkSession

    from .config import Config


def demo_row_filter(spark: SparkSession, cfg: Config) -> dict:
    """Apply a row filter on ``oms_objects.marking``, show its effect, then drop it.

    The workspace has one user, so two identities can't be compared; this shows the mechanism.
    The filter is dropped so dashboards and Genie see every row.

    :return: no facts (empty dict).
    """
    spark.sql(f"""
      CREATE OR REPLACE FUNCTION {cfg.s}.marking_filter(marking STRING)
      RETURN marking = 'OPEN' OR is_account_group_member('admins')""")
    total = spark.table(f"{cfg.s}.oms_objects").count()
    spark.sql(
        f"ALTER TABLE {cfg.s}.oms_objects SET ROW FILTER {cfg.s}.marking_filter ON (marking)"
    )
    print(
        f"rows visible with the filter: {spark.table(f'{cfg.s}.oms_objects').count()} of {total}"
    )
    spark.sql(f"ALTER TABLE {cfg.s}.oms_objects DROP ROW FILTER")
    return {}


def hero_map(spark: SparkSession, cfg: Config, object_id: str) -> folium.Map:
    """Map of one object (red marker) and its ``AUTO``-associated observations (circles)."""
    import folium

    obj = spark.sql(
        f"SELECT * FROM {cfg.s}.oms_objects WHERE object_id = '{object_id}'"
    ).first()
    pts = spark.sql(f"""
      SELECT b.obs_id, b.lat, b.lon, b.reported_type, b.producer, d.match_prob
      FROM {cfg.s}.silver_model_decisions d JOIN {cfg.s}.bronze_observations b ON b.obs_id = d.obs_id
      WHERE d.matched_object_id = '{object_id}' AND d.decision = 'AUTO'""").collect()
    m = folium.Map(location=[obj.lat, obj.lon], zoom_start=15)
    folium.Marker(
        [obj.lat, obj.lon],
        tooltip=f"{obj.object_type} {obj.designator} ({obj.object_id})",
        icon=folium.Icon(color="red"),
    ).add_to(m)
    for pt in pts:
        folium.CircleMarker(
            [pt.lat, pt.lon],
            radius=6,
            tooltip=f"{pt.obs_id} | {pt.reported_type} | {pt.producer} | p={pt.match_prob}",
        ).add_to(m)
    return m


def create_dashboard_views(spark: SparkSession, cfg: Config) -> dict:
    """Create ``dash_decisions``, ``dash_object_summary`` and ``dash_map_points``.

    :return: no facts (empty dict).
    """
    s = cfg.s
    spark.sql(f"""
    CREATE OR REPLACE VIEW {s}.dash_decisions AS
    SELECT decision, count(*) AS observations,
           round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct, round(avg(match_prob), 3) AS avg_match_prob
    FROM {s}.silver_model_decisions GROUP BY decision""")
    spark.sql(f"""
    CREATE OR REPLACE VIEW {s}.dash_object_summary AS
    SELECT object_type, source, status, count(*) AS objects, sum(obs_count) AS linked_observations
    FROM {s}.oms_objects GROUP BY object_type, source, status""")
    spark.sql(f"""
    CREATE OR REPLACE VIEW {s}.dash_map_points AS
    SELECT 'object' AS kind, object_id AS id, lat, lon, object_type AS label, status AS detail,
           CAST(NULL AS DOUBLE) AS match_prob
    FROM {s}.oms_objects
    UNION ALL
    SELECT 'observation', b.obs_id, b.lat, b.lon, coalesce(b.reported_type, 'unidentified'), d.decision, d.match_prob
    FROM {s}.silver_model_decisions d JOIN {s}.bronze_observations b ON b.obs_id = d.obs_id""")
    return {}
