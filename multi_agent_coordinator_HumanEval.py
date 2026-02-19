"""Refactored HumanEval coordinator runner."""

from __future__ import annotations

import random

from ma_llm.data_io import read_data_code
from ma_llm.pipelines import aggregation_dyn_arch as _aggregation_dyn_arch
from ma_llm.pipelines import chain_of_agents as _chain_of_agents


def aggregation_dyn_arch(model, N, adverse, n_pairs, q_and_subq, weights=None):
    """Compatibility wrapper for HumanEval dynamic architecture."""

    return _aggregation_dyn_arch(
        model=model,
        n_agents=N,
        n_adversarial=adverse,
        n_pairs=n_pairs,
        dataset=q_and_subq,
        task="code",
        output_file="5agents_3adv_6conn_code_refactored.json",
        weights=weights,
    )


def chain_of_agents(scored_agents, q_and_a):
    """Compatibility wrapper for HumanEval chain architecture."""

    return _chain_of_agents(
        scored_agents=scored_agents,
        dataset=q_and_a,
        task="code",
        sorted_arg=True,
        output_file="5agents_3adv_chain_code_refactored.json",
    )


if __name__ == "__main__":
    num_agents = 5
    num_adv = 3
    num_links = 6

    questions, solutions, tests, task_ids = read_data_code(file_path="./data/HumanEval.jsonl")
    q_and_a_list = list(zip(questions, solutions, tests, task_ids))

    sampled = random.sample(q_and_a_list, min(100, len(q_and_a_list)))
    scored_agents = aggregation_dyn_arch("mistral:7b", num_agents, num_adv, num_links, sampled)

    chain_of_agents(scored_agents, q_and_a_list)
