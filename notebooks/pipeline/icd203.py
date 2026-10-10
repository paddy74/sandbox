"""ICD 203 expressions of uncertainty: the likelihood scale and analytic confidence levels.

A machine result keeps its number and is shown as the likelihood term for its range; an
analyst enters a term, never a number. Likelihood and confidence are separate judgements and
are never combined in one statement. In CCO terms a likelihood is a Probability Measurement
Information Content Entity (ont00000692); confidence has no CCO class (local extension).
"""

from __future__ import annotations

# (rank, term, low, high) from ICD 203. Adjacent ranges share an endpoint; here each band
# includes its lower bound, and scores below 1% or above 99% fall in the outer bands.
LIKELIHOOD = [
    (1, "Almost no chance", 0.01, 0.05),
    (2, "Very unlikely", 0.05, 0.20),
    (3, "Unlikely", 0.20, 0.45),
    (4, "Roughly even chance", 0.45, 0.55),
    (5, "Likely", 0.55, 0.80),
    (6, "Very likely", 0.80, 0.95),
    (7, "Almost certain", 0.95, 0.99),
]
TERMS = [term for _, term, _, _ in LIKELIHOOD]
RANK = {term: rank for rank, term, _, _ in LIKELIHOOD}

# Analyst confidence in a judgement: the quality of the sources and reasoning behind it.
CONFIDENCE = ["High", "Moderate", "Low"]


def likelihood_term(p: float) -> str:
    """ICD 203 likelihood term for a probability between 0 and 1."""
    for _, term, _, high in LIKELIHOOD[:-1]:
        if p < high:
            return term
    return TERMS[-1]


def likelihood_sql(expr: str) -> str:
    """Spark SQL ``CASE`` giving the likelihood term for a probability expression."""
    whens = " ".join(
        f"WHEN {expr} < {high} THEN '{term}'" for _, term, _, high in LIKELIHOOD[:-1]
    )
    return f"CASE WHEN {expr} IS NULL THEN NULL {whens} ELSE '{TERMS[-1]}' END"


def rank_sql(expr: str) -> str:
    """Spark SQL ``CASE`` giving the rank (1 to 7) of a likelihood term expression; NULL if unknown."""
    whens = " ".join(f"WHEN '{term}' THEN {rank}" for term, rank in RANK.items())
    return f"CASE {expr} {whens} END"
