"""Refactored post-hoc judge script."""

from __future__ import annotations

import json

from ma_llm.judge_eval import judge_evaluation


def run_judge(model, num_agents, num_adv, links, input_file, output_file):
    """Load saved run results and evaluate them using the refactored judge flow."""

    with open(input_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return judge_evaluation(
        model=model,
        n_agents=num_agents,
        n_adversarial=num_adv,
        links=links,
        solutions_payload=data,
        output_file=output_file,
    )


if __name__ == "__main__":
    run_judge(
        model="qwen2.5:7b",
        num_agents=5,
        num_adv=3,
        links=6,
        input_file="5agents_3adv_MATH_chain_qwen2.57b.json",
        output_file="eval_MATH_chain_refactored.json",
    )
