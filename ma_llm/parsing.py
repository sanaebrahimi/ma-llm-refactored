"""Helpers for extracting scores and answers from model outputs."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


def extract_fraction_or_float_values(text: str) -> List[float]:
    """Extract numeric values (fractions and decimals) from text."""

    fractions = re.findall(r"[\d]+/[\\\d]+|[-+]?\d*\.?\d+", text)
    values: List[float] = []
    for token in fractions:
        try:
            values.append(float(Fraction(token)))
        except Exception:
            continue
    return values


def clamp_reward(score: float) -> float:
    """Clamp reward to [-1.0, 1.0]."""

    return max(-1.0, min(1.0, score))


def parse_numeric_score(raw_score: object) -> float:
    """Parse scalar reward value from judge output."""

    if isinstance(raw_score, (float, int)):
        return clamp_reward(float(raw_score))

    if isinstance(raw_score, str):
        match = re.search(r"[-+]?\d*\.?\d+", raw_score)
        if match:
            return clamp_reward(float(match.group(0)))

    return 0.0


def extract_answer_and_score(
    text: str,
    solution: str,
    choices_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], float]:
    """Extract a multiple-choice answer and compare against a reference answer.

    Parameters
    ----------
    text:
        Model output.
    solution:
        Expected answer letter such as ``A``.
    choices_map:
        Optional mapping from answer text to answer letter.
    """

    cleaned = re.sub(r"</?OUTPUT>", "", text, flags=re.IGNORECASE).strip()
    mcq_match = re.search(r"\b([A-D])\)?\b", cleaned, flags=re.IGNORECASE)

    if mcq_match:
        extracted = mcq_match.group(1).upper()
        return extracted, 1.0 if extracted == solution else -1.0

    if choices_map:
        for answer_text, letter in choices_map.items():
            if answer_text in cleaned:
                return letter, 1.0 if letter == solution else -1.0

    return None, 0.0
