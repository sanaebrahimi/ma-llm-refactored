"""Agent orchestration helpers for communication and response collection."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .clients import GPTClient, OllamaClient


def agent_initialize(
    n_agents: int,
    n_adversarial: int,
    model: str,
    regular_system_prompt: str,
    adversarial_system_prompt: str,
    host: str = "localhost:11434",
    backend: str = "ollama",
) -> Tuple[List[object], List[object]]:
    """Create regular and adversarial agent sets."""

    if n_adversarial > n_agents:
        raise ValueError("n_adversarial must be <= n_agents")

    if backend == "openai":
        adversaries = [
            GPTClient(adversarial_system_prompt, model=model, presence_penalty=0.0, frequency_penalty=0.0)
            for _ in range(n_adversarial)
        ]
        regular_agents = [
            GPTClient(regular_system_prompt, model=model, presence_penalty=0.0, frequency_penalty=0.0)
            for _ in range(n_agents - n_adversarial)
        ]
        return adversaries, regular_agents

    adversaries = [OllamaClient(adversarial_system_prompt, model, host=host) for _ in range(n_adversarial)]
    regular_agents = [OllamaClient(regular_system_prompt, model, host=host) for _ in range(n_agents - n_adversarial)]
    return adversaries, regular_agents


def collab_response(
    agent_1_id: str,
    agent_1: OllamaClient,
    agent_2_id: str,
    agent_2: OllamaClient,
    solution: Dict[str, Dict],
) -> Tuple[str, str]:
    """Run one bidirectional collaboration turn between two agents."""

    agent1_res = solution[agent_1_id]["response"]
    agent2_res = solution[agent_2_id]["response"]
    problem = solution["problem"]

    communicate = (
        "Your colleague provided the following response to the question you were asked:\n"
        "The question was: {question}, and your colleague's answer was: {answer}.\n"
        "Based on this, you may revise your answer if you wish; otherwise, repeat your previous response."
    )

    prompt_1 = f"{agent_1.system_template}\n" + communicate.format(question=problem, answer=agent2_res)
    prompt_2 = f"{agent_2.system_template}\n" + communicate.format(question=problem, answer=agent1_res)

    return agent_1.get_response(prompt_1), agent_2.get_response(prompt_2)


def collect_responses(
    solution: Dict[str, Dict],
    agents: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Collect final responses, communication history, and current credibility scores."""

    final_responses: List[str] = []
    communication: List[str] = []
    scores: List[str] = []

    for agent_id in agents:
        if agent_id not in solution:
            continue

        final_response = solution[agent_id]["response"]
        final_responses.append(final_response)
        scores.append(str(solution[agent_id]["score"]))

        history = solution[agent_id]["history"]
        communication.append(f"{agent_id} initial response: {history[0]}")
        for speaker, revised in history[1:]:
            communication.append(f"{agent_id} response after talking to {speaker}: {revised}")
        communication.append(f"{agent_id} final response: {final_response}")

    return final_responses, communication, scores
