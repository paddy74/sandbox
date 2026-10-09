"""Create or update the Genie space from ``demo/genie-examples.md`` via the REST API."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from .config import Config

API = "/api/2.0/genie/spaces"
DOC_SCHEMA = "workspace.obj_resolution_demo"  # schema written in genie-examples.md


def genie_id(*parts: str) -> str:
    """Stable 32-hex ID from the given text, so a re-run sends the same payload."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def parse_genie_md(text: str, schema: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Read the space instructions and examples from ``genie-examples.md``.

    :param schema: ``catalog.schema`` to use in place of the one written in the file.
    :return: (instructions, [(number, question, sql)]).
    :raises ValueError: if the instructions block or any example's ``sql`` block is missing.
    """
    text = text.replace(DOC_SCHEMA, schema)
    instr = re.search(r"## Space instructions.*?```markdown\n(.*?)```", text, re.S)
    examples = [
        (int(n), q.strip(), sql.strip())
        for n, q, sql in re.findall(
            r"^### (\d+)\. (.+?)\n.*?```sql\n(.*?)```", text, re.S | re.M
        )
    ]
    if not instr or len(examples) != len(re.findall(r"^### \d+\. ", text, re.M)):
        raise ValueError(
            "every ### example needs a ```sql block, plus the instructions block"
        )
    return instr.group(1).strip(), examples


def by_id(items: list[dict]) -> list[dict]:
    """Sort by ``id``: the API requires its lists in ID order."""
    return sorted(items, key=lambda x: x["id"])


def sync_space(
    spark: SparkSession,
    cfg: Config,
    title: str = "Object resolution demo",
    sample_questions: tuple[int, ...] = (1, 5, 2),
    warehouse_id: str | None = None,
) -> dict:
    """Create the Genie space, or update the one with the same title.

    Only what the markdown owns is replaced: sample questions, text instructions, example
    SQL and the table list (the tables the examples query). Anything else set in the UI, such
    as hidden columns or joins, is kept. Feasibility: section 11 of ``feasibility_tests.py``.

    :param sample_questions: example numbers shown as starter questions.
    :param warehouse_id: SQL warehouse; ``None`` keeps an existing space's warehouse, or picks
        the first warehouse for a new space.
    :return: cheat-card facts: whether the space was created or updated, and its ID.
    :raises RuntimeError: if several spaces share the title or no warehouse exists.
    """
    from databricks.sdk import WorkspaceClient

    md = cfg.repo_root / "demo" / "genie-examples.md"
    instructions, examples = parse_genie_md(md.read_text(), cfg.s)
    by_number = {n: q for n, q, _ in examples}
    missing = [n for n in sample_questions if n not in by_number]
    assert not missing, f"sample question numbers not in {md.name}: {missing}"
    tables = sorted(
        {
            t
            for _, _, q in examples
            for t in re.findall(rf"{re.escape(cfg.s)}\.(\w+)", q)
        }
    )

    w = WorkspaceClient()
    existing = [
        sp
        for sp in w.api_client.do("GET", API).get("spaces", [])
        if sp.get("title") == title
    ]
    if len(existing) > 1:
        raise RuntimeError(
            f"{len(existing)} Genie spaces titled {title!r}; trash the extras first"
        )
    if existing:
        got = w.api_client.do(
            "GET",
            f"{API}/{existing[0]['space_id']}",
            query={"include_serialized_space": "true"},
        )
        space = json.loads(got["serialized_space"])
    else:
        got, space = {}, {"version": 2}

    space.setdefault("config", {})["sample_questions"] = by_id(
        [
            {"id": genie_id("sample", by_number[n]), "question": [by_number[n]]}
            for n in sample_questions
        ]
    )
    space.setdefault("instructions", {})["text_instructions"] = [
        {"id": genie_id("instructions"), "content": [instructions]}
    ]
    space["instructions"]["example_question_sqls"] = by_id(
        [
            {"id": genie_id("example", q), "question": [q], "sql": [sql]}
            for _, q, sql in examples
        ]
    )
    kept = {
        t["identifier"]: t
        for t in space.setdefault("data_sources", {}).get("tables", [])
    }
    space["data_sources"]["tables"] = [
        kept.get(f"{cfg.s}.{t}", {"identifier": f"{cfg.s}.{t}"}) for t in tables
    ]

    body = {"title": title, "serialized_space": json.dumps(space)}
    if warehouse_id or not existing:
        wh = warehouse_id or next((x.id for x in w.warehouses.list()), None)
        if not wh:
            raise RuntimeError(
                "no SQL warehouse found; create one or pass warehouse_id"
            )
        body["warehouse_id"] = wh
    if existing:
        res = w.api_client.do(
            "PATCH",
            f"{API}/{existing[0]['space_id']}",
            body={**body, "etag": got.get("etag")},
        )
    else:
        res = w.api_client.do("POST", API, body=body)
        print(
            "New space: hide true_object_id (bronze_observations) and dup_of (oms_objects) in the Genie UI once;"
        )
        print("later runs keep that setting.")
    status = f"{'updated' if existing else 'created'} {res['space_id']}"
    print(f"Genie space {status}: {len(examples)} examples, tables {tables}")
    return {"genie_space": status}
