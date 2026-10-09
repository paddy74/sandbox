"""Locations and demo settings shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

SCHEMA = "obj_resolution_demo"


@dataclass(frozen=True)
class Config:
    """Where the demo lives and the thresholds it uses.

    :param catalog: Unity Catalog catalog; the schema and landing volume are created in it.
    """

    catalog: str
    schema: str = SCHEMA
    # match_prob >= auto_t: associate; >= review_t: analyst review; below: nominate.
    auto_t: float = 0.9
    review_t: float = 0.5
    h3_res: int = 8
    max_dist_m: int = 500  # candidate search radius
    ai_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    repo_root: Path = field(default=Path(__file__).resolve().parents[2])

    @classmethod
    def from_spark(cls, spark: SparkSession) -> Config:
        """Config for the session's current catalog."""
        return cls(catalog=spark.sql("SELECT current_catalog()").first()[0])

    @property
    def s(self) -> str:
        """Fully qualified schema, ``catalog.schema``."""
        return f"{self.catalog}.{self.schema}"

    @property
    def vol(self) -> str:
        """Landing volume path."""
        return f"/Volumes/{self.catalog}/{self.schema}/landing"

    @property
    def landing(self) -> dict[str, str]:
        """Landing folder per source; ``staged`` holds observation files not yet delivered."""
        return {
            name: f"{self.vol}/{name}"
            for name in ["observations", "staged", "tasking", "reports"]
        }


def hav_sql(lat1: str, lon1: str, lat2: str, lon2: str) -> str:
    """Haversine distance in metres as a Spark SQL expression over the given columns."""
    return (
        f"2*6371000*asin(sqrt(pow(sin(radians({lat1}-{lat2})/2),2)"
        f" + cos(radians({lat1}))*cos(radians({lat2}))*pow(sin(radians({lon1}-{lon2})/2),2)))"
    )
