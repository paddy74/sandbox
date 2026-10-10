"""Pick the three demo storyline objects from the data and store them in ``demo_heroes``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .icd203 import RANK, rank_sql
from .synthetic import GENERIC_LABELS

if TYPE_CHECKING:
    from pyspark.sql import Row, SparkSession

    from .config import Config


def select_heroes(spark: SparkSession, cfg: Config) -> dict:
    """Write ``demo_heroes``.

    **A** a clean auto-associate: an Armored Fighting Vehicle with the most corroborating
    ``AUTO`` observations. **B** a ``REVIEW`` observation sitting between two seeded look-alike
    twins (a Truck if possible, top score nearest 0.65). **C** a nominated cluster built mostly
    from weak observations: generic or missing type, or a source likelihood of at most roughly
    even chance. Fails if the data has no candidate for a hero.

    :return: cheat-card facts: the three heroes.
    """
    s = cfg.s
    weak_list = ", ".join(f"'{t}'" for t in GENERIC_LABELS)
    even = RANK["Roughly even chance"]  # a source likelihood at or below this is weak
    a = spark.sql(f"""
      SELECT d.matched_object_id AS object_id, count(*) n_obs, count(DISTINCT b.producer) n_producers,
             round(avg(d.match_prob), 3) AS avg_prob
      FROM {s}.silver_model_decisions d
      JOIN {s}.bronze_observations b ON d.obs_id = b.obs_id
      JOIN {s}.oms_objects o ON o.object_id = d.matched_object_id
      WHERE d.decision = 'AUTO' AND o.object_type = 'Armored Fighting Vehicle' AND o.source = 'OMS'
      GROUP BY d.matched_object_id HAVING count(*) >= 3
      ORDER BY n_producers DESC, n_obs DESC, avg_prob DESC LIMIT 1""").first()
    assert a is not None, "no Armored Fighting Vehicle with >= 3 AUTO observations"

    b = spark.sql(f"""
      WITH r AS (SELECT obs_id, object_id, match_prob,
                        row_number() OVER (PARTITION BY obs_id ORDER BY match_prob DESC) rk
                 FROM {s}.silver_scored_pairs)
      SELECT a.obs_id, a.object_id AS object_id, b.object_id AS alt_object_id,
             round(a.match_prob, 3) AS p1, round(b.match_prob, 3) AS p2, x.object_type
      FROM r a JOIN r b ON a.obs_id = b.obs_id AND a.rk = 1 AND b.rk = 2
      JOIN {s}.oms_objects x ON x.object_id = a.object_id
      JOIN {s}.silver_model_decisions d ON d.obs_id = a.obs_id AND d.decision = 'REVIEW'
      JOIN {s}.seed_twins t ON (t.original_id = a.object_id AND t.twin_id = b.object_id)
                            OR (t.twin_id = a.object_id AND t.original_id = b.object_id)
      WHERE b.match_prob >= 0.3 AND NOT t.is_duplicate
      ORDER BY (x.object_type = 'Truck') DESC, abs(a.match_prob - 0.65) ASC LIMIT 1""").first()
    assert b is not None, (
        "no REVIEW observation sits between two seeded look-alike twins"
    )

    c = spark.sql(f"""
      SELECT m.object_id, count(*) n_obs,
             sum(CASE WHEN b.reported_type IS NULL OR b.reported_type IN ({weak_list})
                        OR {rank_sql("b.likelihood")} <= {even} THEN 1 ELSE 0 END) AS weak_obs
      FROM {s}.silver_nom_members m JOIN {s}.bronze_observations b ON m.obs_id = b.obs_id
      GROUP BY m.object_id HAVING count(*) >= 3
      ORDER BY weak_obs / count(*) DESC, n_obs DESC, m.object_id LIMIT 1""").first()
    assert c is not None, "no nominated cluster with >= 3 observations"

    heroes = [
        ("A", a.object_id, None, None,
         f"{a.n_obs} AUTO observations from {a.n_producers} producer(s), mean match_prob {a.avg_prob}"),
        ("B", b.object_id, b.obs_id, b.alt_object_id,
         f"{b.object_type}; REVIEW obs between twins: match_prob {b.p1} vs runner-up {b.p2}"),
        ("C", c.object_id, None, None,
         f"nominated from {c.n_obs} observations, {c.weak_obs} with generic or missing type or a weak source likelihood"),
    ]  # fmt: skip
    spark.createDataFrame(
        heroes,
        "hero STRING, object_id STRING, obs_id STRING, alt_object_id STRING, detail STRING",
    ).write.mode("overwrite").saveAsTable(f"{s}.demo_heroes")
    return {
        "heroes": {
            h[0]: {"object_id": h[1], "obs_id": h[2], "alt": h[3], "detail": h[4]}
            for h in heroes
        }
    }


def load_heroes(spark: SparkSession, cfg: Config) -> dict[str, Row]:
    """Heroes from ``demo_heroes``, keyed ``"A"``, ``"B"``, ``"C"``; run ``select_heroes`` first."""
    rows = {r.hero: r for r in spark.table(f"{cfg.s}.demo_heroes").collect()}
    if set(rows) != {"A", "B", "C"}:
        raise RuntimeError(
            f"demo_heroes has {sorted(rows)}; run heroes.select_heroes first"
        )
    return rows
