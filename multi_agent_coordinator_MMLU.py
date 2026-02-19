"""Refactored MMLU coordinator runner."""

from __future__ import annotations

import random

from ma_llm.data_io import read_mmlu_mathematics
from ma_llm.pipelines import aggregation_dyn_arch as _aggregation_dyn_arch
from ma_llm.pipelines import chain_of_agents as _chain_of_agents


def aggregation_dyn_arch(model, N, adverse, n_pairs, q_and_subq, weights=None):
    """Compatibility wrapper for MMLU dynamic architecture."""

    return _aggregation_dyn_arch(
        model=model,
        n_agents=N,
        n_adversarial=adverse,
        n_pairs=n_pairs,
        dataset=q_and_subq,
        task="mmlu",
        output_file="5agents_3adv_4conn_mmlu_refactored.json",
        weights=weights,
    )


def chain_of_agents(scored_agents, q_and_a, sorted_arg=True, name="x.json"):
    """Compatibility wrapper for MMLU chain architecture."""

    return _chain_of_agents(
        scored_agents=scored_agents,
        dataset=q_and_a,
        task="mmlu",
        sorted_arg=sorted_arg,
        output_file=name,
    )


if __name__ == "__main__":
    num_agents = 5
    num_adv = 3
    num_links = 4

    questions, choices, solutions = read_mmlu_mathematics(
        file_path="./data/MMLU/high_school_statistics_test.csv"
    )
    q_and_a_list = list(zip(questions, choices, solutions))

    sampled_q_and_a = random.sample(q_and_a_list, min(50, len(q_and_a_list)))
    scored_agents = aggregation_dyn_arch("llama3.2", num_agents, num_adv, num_links, sampled_q_and_a)

    sampled_chain = random.sample(q_and_a_list, min(50, len(q_and_a_list)))
    chain_of_agents(scored_agents, sampled_chain, sorted_arg=True, name="5agents_3adv_mmlu_chain_refactored.json")
