"""Refactored disconnected multi-agent runner (embedding aggregation mode)."""

from __future__ import annotations

import random

from ma_llm.data_io import read_gsm8k
from ma_llm.embedding_pipeline import aggregation


if __name__ == "__main__":
    num_agents = 5
    num_adv = 3

    questions, solutions = read_gsm8k(file_path="./data/gsm8k.jsonl")
    q_and_a_list = list(zip(questions, solutions))
    sampled = random.sample(q_and_a_list, min(100, len(q_and_a_list)))

    aggregation(
        n_agents=num_agents,
        n_adversarial=num_adv,
        dataset=sampled,
        model="llama3.2",
        output_file="5agents_3adv_disconnected_gsm8k_refactored.json",
    )
