"""Refactored MATH/GSM8K coordinator runner."""

from __future__ import annotations

import random

from ma_llm.data_io import read_gsm8k
from ma_llm.pipelines import aggregation_dyn_arch as _aggregation_dyn_arch
from ma_llm.pipelines import chain_of_agents as _chain_of_agents


def aggregation_dyn_arch(model, N, adverse, n_pairs, q_and_subq, weights=None):
    """Compatibility wrapper for math dynamic architecture."""

    return _aggregation_dyn_arch(
        model=model,
        n_agents=N,
        n_adversarial=adverse,
        n_pairs=n_pairs,
        dataset=q_and_subq,
        task="math",
        output_file="5agents_3adv_6conn_math_refactored.json",
        weights=weights,
        judge_backend="ollama",
        judge_model=model,
    )


def chain_of_agents(scored_agents, q_and_a, sorted_arg=True, file_n="results_math_chain.json"):
    """Compatibility wrapper for math chain architecture."""

    return _chain_of_agents(
        scored_agents=scored_agents,
        dataset=q_and_a,
        task="math",
        sorted_arg=sorted_arg,
        output_file=file_n,
        judge_backend="ollama",
        judge_model="llama3.2",
    )


if __name__ == "__main__":
    num_agents = 5
    num_adv = 3
    num_links = 6

    questions, solutions = read_gsm8k(file_path="./data/gsm8k.jsonl")
    q_and_a_list = list(zip(questions, solutions))

    sampled_q_and_a = random.sample(q_and_a_list, min(50, len(q_and_a_list)))
    scored_agents = aggregation_dyn_arch("qwen2.5:7b", num_agents, num_adv, num_links, sampled_q_and_a)

    sampled_chain = random.sample(q_and_a_list, min(50, len(q_and_a_list)))
    chain_of_agents(scored_agents, sampled_chain, sorted_arg=True, file_n="5agents_3adv_math_chain_refactored.json")
