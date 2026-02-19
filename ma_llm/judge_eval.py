"""Post-hoc evaluation pipeline for previously saved multi-agent runs."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .clients import JudgeClient, JudgeOllamaClient, OllamaClient
from .data_io import write_data
from .orchestration import collect_responses
from .templates import JUDGE_MATH


def judge_evaluation(
    model: str,
    n_agents: int,
    n_adversarial: int,
    links: int,
    solutions_payload: Iterable[dict],
    output_file: str,
    judge_backend: str = "openai",
) -> List[dict]:
    """Evaluate historical runs by re-aggregating agent responses and scoring output.

    Parameters
    ----------
    model:
        Local model used by coordinator for re-aggregation.
    n_agents:
        Number of participating agents.
    n_adversarial:
        Count of adversarial agents, recorded for metadata.
    links:
        Communication links, recorded for metadata.
    solutions_payload:
        Iterable of existing per-question solution dictionaries.
    output_file:
        Output JSON filename.
    """

    judge = JudgeOllamaClient(JUDGE_MATH, model=model) if judge_backend == "ollama" else JudgeClient(JUDGE_MATH)
    coordinator = OllamaClient(
        "You will receive a question and answers from multiple agents. Use only those responses to produce the final answer.",
        model,
    )

    results: List[dict] = []

    for item in solutions_payload:
        question = item.get("problem", "")
        reference = item.get("solution", "")

        solution: Dict[str, object] = {"problem": question, "solution": reference}
        agent_ids = [f"agent {idx}" for idx in range(n_agents)]

        for agent_id in agent_ids:
            if agent_id not in item:
                continue
            original = item[agent_id]
            solution[agent_id] = {
                "response": original.get("response", ""),
                "history": original.get("history", [original.get("response", "")]),
                "score": original.get("score", 0),
            }

        responses, _, _ = collect_responses(solution, agent_ids)
        final_answer = coordinator.get_response(f"Question: {question}\nResponses: {', '.join(responses)}")
        solution["coordinator"] = {"response": final_answer}

        prompt = (
            f"Question: {question}\n"
            f"Team's Proposed Solution: {final_answer}\n"
            f"Reference Solution: {reference}\n"
        )
        solution["reward"] = judge.get_score(prompt)
        results.append(solution)

    results.append({"Agents": n_agents, "Adv": n_adversarial, "links": links, "model": model})
    write_data(file_name=output_file, solutions=results)
    return results
