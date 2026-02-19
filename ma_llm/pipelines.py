"""Core multi-agent pipelines (dynamic graph and chain architectures)."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .clients import GPTClient, JudgeClient, JudgeOllamaClient, OllamaClient
from .data_io import write_data
from .orchestration import agent_initialize, collab_response, collect_responses
from .parsing import extract_answer_and_score
from .scoring import calculate_contribution, update_weights
from .templates import TaskTemplates, get_templates


def _format_question_block(question: str, choices: Optional[Sequence[str]] = None) -> str:
    """Format a question block, including MCQ choices when provided."""

    if not choices:
        return f"Question: {question}"
    return (
        f"Question: {question}\n"
        f"Choices: A) {choices[0]}\n B) {choices[1]}\n C) {choices[2]}\n D) {choices[3]}"
    )


def _agent_prompt(task: str, question: str, choices: Optional[Sequence[str]] = None) -> str:
    if task == "code":
        return f"Coding snippet: {question}\n"
    return _format_question_block(question, choices)


def _coordinator_prompt(
    task: str,
    question: str,
    responses: Sequence[str],
    scores: Optional[Sequence[str]] = None,
    choices: Optional[Sequence[str]] = None,
) -> str:
    """Build coordinator input text for one round."""

    question_block = _format_question_block(question, choices)
    response_block = ", ".join(responses)
    if scores is None:
        return f"{question_block}\nResponses: {response_block}."
    return (
        f"{question_block}\nResponses: {response_block}.\n"
        f"Credibility Scores in the order of responses: {', '.join(scores)}"
    )


def _judge_prompt(
    task: str,
    question: str,
    reference_solution: str,
    team_solution: str,
    choices: Optional[Sequence[str]] = None,
    test: Optional[str] = None,
) -> str:
    """Build judge prompt text for one round."""

    if task == "qa":
        return (
            f"Question: {question}\n"
            f"Team's Proposed Solution: {team_solution}\n"
        )

    if task == "code":
        return (
            f"Code snippet: {question}\n"
            f"Reference solution: {reference_solution}\n"
            f"Team's Proposed Solution: {team_solution}\n"
            f"Unit tests: {test}\n"
        )

    if task == "mmlu" and choices is not None:
        return (
            f"Question: {question}\n"
            f"Choices: A) {choices[0]}\n B) {choices[1]}\n C) {choices[2]}\n D) {choices[3]}\n"
            f"Team's Proposed Solution: {team_solution}\n"
            f"Reference Solution: Option {reference_solution}\n"
        )

    return (
        f"Question: {question}\n"
        f"Team's Proposed Solution: {team_solution}\n"
        f"Reference Solution: {reference_solution}\n"
    )


def _reward_for_round(
    task: str,
    judge,
    question: str,
    reference_solution: str,
    team_solution: str,
    choices: Optional[Sequence[str]] = None,
    test: Optional[str] = None,
):
    """Compute round reward using direct parsing when available, else judge score."""

    if task == "mmlu" and choices is not None:
        choices_map = {choices[0]: "A", choices[1]: "B", choices[2]: "C", choices[3]: "D"}
        _, direct = extract_answer_and_score(team_solution, reference_solution, choices_map)
        if direct != 0:
            return direct

    prompt = _judge_prompt(task, question, reference_solution, team_solution, choices=choices, test=test)
    return judge.get_score(prompt)


def _build_judge(task: str, templates: TaskTemplates, judge_backend: str, model: str):
    """Instantiate judge client based on backend choice."""

    if judge_backend == "ollama":
        return JudgeOllamaClient(templates.judge, model=model)
    return JudgeClient(templates.judge)


def aggregation_dyn_arch(
    model: str,
    n_agents: int,
    n_adversarial: int,
    n_pairs: int,
    dataset: Iterable[tuple],
    task: str,
    output_file: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    ollama_host: str = "localhost:11434",
    judge_backend: str = "openai",
    judge_model: Optional[str] = None,
    llm_backend: str = "ollama",
    llm_model: Optional[str] = None,
):
    """Run the dynamic communication architecture used in the original scripts."""

    templates = get_templates(task)
    judge = _build_judge(task, templates, judge_backend, judge_model or model)

    if weights is None:
        current_weights = np.array([1.0 / n_agents] * n_agents)
    else:
        current_weights = np.array(weights, dtype=float)

    runtime_model = llm_model or model
    adversaries, regular_agents = agent_initialize(
        n_agents,
        n_adversarial,
        runtime_model,
        regular_system_prompt=templates.system,
        adversarial_system_prompt=templates.adversarial_system,
        host=ollama_host,
        backend=llm_backend,
    )
    agents = regular_agents + adversaries
    if llm_backend == "openai":
        coordinator = GPTClient(
            templates.coordinator_weighted,
            model=runtime_model,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
    else:
        coordinator = OllamaClient(templates.coordinator_weighted, runtime_model, host=ollama_host)

    results: List[Dict] = []
    agents_w_id: Dict[str, OllamaClient] = {}

    for item in dataset:
        question = item[0]
        choices = item[1] if task == "mmlu" else None
        reference = item[2] if task == "mmlu" else item[1]
        test = item[2] if task == "code" else None
        task_id = item[3] if task == "code" else None

        solution: Dict[str, object] = {"problem": question}
        if task == "qa":
            solution["subproblem"] = reference
        else:
            solution["solution"] = reference
        if task == "code":
            solution["test"] = test
            solution["task_id"] = task_id

        list_of_agents = [f"agent {idx}" for idx in range(n_agents)]
        prompt = _agent_prompt(task, question, choices)

        for idx in range(n_agents):
            agent_id = list_of_agents[idx]
            response = agents[idx].get_response(prompt)
            agent_subproblem = reference[idx] if task == "qa" and isinstance(reference, list) else None
            solution[agent_id] = {
                "response": response,
                "history": [response],
                "score": float(current_weights[idx]),
            }
            if agent_subproblem is not None:
                solution[agent_id]["subproblem"] = agent_subproblem
            agents_w_id[agent_id] = agents[idx]

        for _ in range(n_pairs):
            id_1, id_2 = np.random.choice(list_of_agents, size=2, replace=False)
            new_res_1, new_res_2 = collab_response(id_1, agents_w_id[id_1], id_2, agents_w_id[id_2], solution)
            solution[id_1]["response"] = new_res_1
            solution[id_1]["history"].append((id_2, new_res_1))
            solution[id_2]["response"] = new_res_2
            solution[id_2]["history"].append((id_1, new_res_2))

        responses, communication, credibility_scores = collect_responses(solution, list_of_agents)

        coordinator.system_template = templates.coordinator_weighted
        final_answer = coordinator.get_response(
            _coordinator_prompt(task, question, responses, credibility_scores, choices=choices)
        )

        solution["coordinator"] = {"response": final_answer}
        reward = _reward_for_round(
            task,
            judge,
            question,
            reference,
            final_answer,
            choices=choices,
            test=test,
        )

        # Preserve QA behavior from the original script: include nonscored coordinator baseline.
        if task == "qa":
            coordinator.system_template = templates.coordinator_nonscored
            nonscored_answer = coordinator.get_response(
                _coordinator_prompt(task, question, responses, None, choices=choices)
            )
            solution["coordinator"]["nonscored response"] = nonscored_answer
            solution["nonscored reward"] = _reward_for_round(
                task,
                judge,
                question,
                reference,
                nonscored_answer,
                choices=choices,
                test=test,
            )

        shapley_vals = calculate_contribution(judge, communication, responses, final_answer)
        current_weights, parsed_reward = update_weights(reward, credibility_scores, shapley_vals)

        solution["weights"] = current_weights.tolist()
        solution["shapley"] = list(shapley_vals.values())
        solution["reward"] = parsed_reward

        results.append(solution)
        if output_file:
            write_data(file_name=output_file, solutions=results)

    scored_agents = [
        (float(current_weights[idx]), agents_w_id[f"agent {idx}"], f"agent {idx}")
        for idx in range(n_agents)
    ]
    scored_agents.append((0.0, coordinator, "coordinator"))
    return scored_agents


def chain_of_agents(
    scored_agents: List[Tuple],
    dataset: Iterable[tuple],
    task: str,
    sorted_arg: bool = True,
    output_file: Optional[str] = None,
    judge_backend: str = "openai",
    judge_model: str = "gpt-4o-mini",
):
    """Run chain architecture where each agent sees the previous agent's output."""

    if not scored_agents:
        return 0

    templates = get_templates(task)
    judge = _build_judge(task, templates, judge_backend, judge_model)

    *agent_entries, coordinator_entry = scored_agents
    coordinator = coordinator_entry[1]

    ranked = sorted(agent_entries, key=lambda x: float(x[0]), reverse=True) if sorted_arg else list(agent_entries)
    n_agents = len(ranked)
    weights = [float(agent[0]) for agent in ranked]

    results: List[Dict] = []

    for item in dataset:
        question = item[0]
        choices = item[1] if task == "mmlu" else None
        reference = item[2] if task == "mmlu" else item[1]
        test = item[2] if task == "code" else None
        task_id = item[3] if task == "code" else None

        solution: Dict[str, object] = {"problem": question}
        if task == "qa":
            solution["subproblem"] = reference
        else:
            solution["solution"] = reference
        if task == "code":
            solution["test"] = test
            solution["task_id"] = task_id

        # Agent 0 answers directly.
        first_id, first_agent = ranked[0][2], ranked[0][1]
        first_response = first_agent.get_response(_agent_prompt(task, question, choices))
        solution[first_id] = {"response": first_response, "history": [first_response], "score": weights[0]}

        for idx in range(1, n_agents):
            current_id = ranked[idx][2]
            prev_id = ranked[idx - 1][2]
            question_block = _agent_prompt(task, question, choices)
            prompt = (
                f"Here is the prompt you need to answer:\n{question_block}\n"
                f"Your colleague previously responded with:\n{solution[prev_id]['response']}\n"
                "You may use or refine this information as needed to improve your answer."
            )
            response = ranked[idx][1].get_response(prompt)
            solution[current_id] = {
                "response": response,
                "history": [(prev_id, response)],
                "score": weights[idx],
            }

        list_of_agents = [f"agent {idx}" for idx in range(n_agents)]
        responses, communication, credibility_scores = collect_responses(solution, list_of_agents)

        if sorted_arg:
            coordinator.system_template = templates.coordinator_weighted
            coordinator_prompt = _coordinator_prompt(task, question, responses, credibility_scores, choices=choices)
        else:
            coordinator.system_template = templates.coordinator_nonscored
            coordinator_prompt = _coordinator_prompt(task, question, responses, None, choices=choices)

        final_answer = coordinator.get_response(coordinator_prompt)
        solution["coordinator"] = {"response": final_answer}

        reward = _reward_for_round(
            task,
            judge,
            question,
            reference,
            final_answer,
            choices=choices,
            test=test,
        )

        if sorted_arg:
            shapley_vals = calculate_contribution(judge, communication, responses, final_answer)
            updated_weights, parsed_reward = update_weights(reward, credibility_scores, shapley_vals)
            solution["weights"] = updated_weights.tolist()
            solution["shapley"] = list(shapley_vals.values())
            solution["reward"] = parsed_reward
        else:
            solution["reward"] = reward

        results.append(solution)
        if output_file:
            write_data(file_name=output_file, solutions=results)

    return 0
