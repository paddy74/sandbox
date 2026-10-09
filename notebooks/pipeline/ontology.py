"""CCO ontology: flatten the class hierarchy and resolve the demo's type labels to IRIs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .synthetic import DESIGNATOR_IRI, GENERIC, GENERIC_LABELS, LEAVES

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from .config import Config

CCO_FILE = "CommonCoreOntologiesMerged.ttl"


def find_cco(cfg: Config) -> Path:
    """The ontology file from the repo, else from the landing volume.

    :raises FileNotFoundError: if neither copy exists or the file is truncated.
    """
    candidates = [
        cfg.repo_root / "data" / "ontology" / CCO_FILE,
        Path(cfg.vol) / "ontology" / CCO_FILE,
    ]
    found = next(
        (p for p in candidates if p.exists() and p.stat().st_size > 500_000), None
    )
    if found is None:
        raise FileNotFoundError(f"No valid {CCO_FILE} (>500 KB) in: {candidates}")
    return found


def load_cco(spark: SparkSession, cfg: Config) -> dict:
    """Write ``cco_classes``, ``type_iri`` and ``type_ancestors``.

    ``type_ancestors`` (which generic label a specific type falls under) drives the type
    score in ``model.build_candidates``, so type compatibility comes from the ontology, not a
    hand-built table. Fails if a demo label is missing or ambiguous in CCO, or if CCO does not
    place a type under the generic label the generator reports for it.

    :return: no facts (empty dict).
    """
    from rdflib import OWL, RDF, RDFS, Graph, URIRef

    path = find_cco(cfg)
    g = Graph()
    g.parse(path, format="turtle")
    classes = {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    parents = {
        c: [p for p in g.objects(c, RDFS.subClassOf) if isinstance(p, URIRef)]
        for c in classes
    }
    memo: dict = {}

    def ancestors(c, seen=()):
        """All superclasses of ``c`` (transitive), skipping cycles."""
        if c in memo:
            return memo[c]
        out = set()
        for p in parents.get(c, []):
            if p in seen:
                continue
            out |= {p} | ancestors(p, seen + (c,))
        memo[c] = out
        return out

    rows = [
        (str(c), str(g.value(c, RDFS.label) or ""), [str(p) for p in parents[c]],
         sorted(str(a) for a in ancestors(c)))
        for c in classes
    ]  # fmt: skip
    spark.createDataFrame(
        rows,
        "iri STRING, label STRING, parent_iris ARRAY<STRING>, ancestor_iris ARRAY<STRING>",
    ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{cfg.s}.cco_classes"
    )
    print(f"triples={len(g)} classes={len(classes)}")

    labels = LEAVES + GENERIC_LABELS
    spark.createDataFrame(
        [(x,) for x in labels], "label STRING"
    ).createOrReplaceTempView("demo_labels")
    spark.sql(f"""
      CREATE OR REPLACE TABLE {cfg.s}.type_iri AS
      SELECT d.label, c.iri FROM demo_labels d JOIN {cfg.s}.cco_classes c ON lower(c.label) = lower(d.label)""")
    cnt = spark.sql(
        f"SELECT label, count(*) n FROM {cfg.s}.type_iri GROUP BY label"
    ).toPandas()
    missing = sorted(set(labels) - set(cnt.label))
    ambiguous = cnt[cnt.n > 1].label.tolist()
    assert not missing and not ambiguous, (
        f"CCO label problems: missing={missing} ambiguous={ambiguous}"
    )

    spark.sql(f"""
      CREATE OR REPLACE TABLE {cfg.s}.type_ancestors AS
      SELECT t.label AS type, a.label AS ancestor
      FROM {cfg.s}.type_iri t
      JOIN {cfg.s}.cco_classes c ON c.iri = t.iri
      JOIN {cfg.s}.type_iri a ON array_contains(c.ancestor_iris, a.iri)""")
    ta = spark.table(f"{cfg.s}.type_ancestors").toPandas()
    pairs = set(zip(ta.type, ta.ancestor, strict=True))
    bad = [(leaf, gen) for leaf, gen in GENERIC.items() if (leaf, gen) not in pairs]
    assert not bad, (
        f"CCO does not place these under the generic label used by the generator: {bad}"
    )
    assert str(g.value(URIRef(DESIGNATOR_IRI), RDFS.label)) == "Arbitrary Identifier", (
        f"{DESIGNATOR_IRI} is not CCO Arbitrary Identifier in {path}"
    )
    return {}
