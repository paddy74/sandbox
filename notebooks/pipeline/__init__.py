"""Stages of the object resolution pipeline, run in order by ``notebooks/generate_data.py``.

Each stage takes ``(spark, cfg)``, reads its inputs from tables and returns cheat-card facts,
so it can run on its own once the upstream tables exist.
"""
