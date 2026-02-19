"""Disconnected (non-communication) aggregation pipeline using embedding voting."""

from __future__ import annotations

from itertools import chain, combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .clients import JudgeClient, OllamaClient
from .data_io import write_data
from .parsing import parse_numeric_score
from .scoring import calculate_cosine
from .templates import CODE_TEMPLATES, MATH_TEMPLATES

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


def subsets_element(s: Sequence[str], element: str) -> List[List[str]]:
    """Return all subsets of ``s`` (size > 1) that include ``element``."""

    if element not in s:
        raise ValueError("The specified element is not in the set.")
    all_subsets = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return [list(subset) for subset in all_subsets if element in subset and len(subset) > 1]


def subset_no_element(s: Sequence[str], element: str) -> List[List[str]]:
    """Return all non-empty subsets of ``s`` that do not include ``element``."""

    if element not in s:
        raise ValueError("The specified element is not in the set.")
    all_subsets = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return [list(subset) for subset in all_subsets if element not in subset and len(subset) > 0]


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Return embedding vector for text via OpenAI embeddings API."""

    if OpenAI is None:
        raise RuntimeError("OpenAI client unavailable. Install `openai` and set OPENAI_API_KEY.")
    client = OpenAI()
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def embedding_collections(sentences: Sequence[str], weights: Sequence[float]):
    """Build embedding and weight maps for sentence-level aggregation."""

    embedding_dict: Dict[str, List[float]] = {}
    weights_dict: Dict[Tuple[float, ...], float] = {}

    for idx, sentence in enumerate(sentences):
        tagged = f"agent {idx}: {sentence}"
        vector = get_embedding(tagged)
        embedding_dict[tagged] = vector
        weights_dict[tuple(vector)] = float(weights[idx])

    return embedding_dict, weights_dict, list(embedding_dict.keys())


def generate_response(sentences: Sequence[str], vectors: Dict[str, List[float]], weights: Dict[Tuple[float, ...], float]):
    """Select response with highest cosine similarity to weighted centroid."""

    if len(sentences) == 1:
        sentence = sentences[0]
        return sentence, vectors[sentence]

    weighted_vectors = [np.array(vectors[s]) * weights[tuple(vectors[s])] for s in sentences]
    average = np.mean(weighted_vectors, axis=0)

    similarity = {
        s: calculate_cosine(average, vectors[s])
        for s in sentences
    }
    best = max(similarity, key=similarity.get)
    return best, vectors[best]


def calculate_shap(sentences: Sequence[str], final_response: str, embedding_dict, weights):
    """Compute Shapley-like contribution values over embedding coalitions."""

    n = len(sentences)
    shapley_values = {sentence: 0.0 for sentence in sentences}
    final_embed = embedding_dict[final_response]

    for sentence in sentences:
        for subset in subset_no_element(sentences, sentence):
            with_sentence = subset + [sentence]
            _, best_with = generate_response(with_sentence, embedding_dict, weights)
            _, best_without = generate_response(subset, embedding_dict, weights)
            marginal = calculate_cosine(best_with, final_embed) - calculate_cosine(best_without, final_embed)
            shapley_values[sentence] += (1 / np.math.factorial(n)) * marginal

        shapley_values[sentence] += (1 / np.math.factorial(n)) * calculate_cosine(embedding_dict[sentence], final_embed)

    return shapley_values


def update_weights(raw_score, weights: Dict[Tuple[float, ...], float], shapley_vals):
    """Update disconnected-graph weights with reward and Shapley contributions."""

    score = parse_numeric_score(raw_score)
    base = list(weights.values())
    updated = [base[idx] * (1 + score * val * 0.1) for idx, val in enumerate(shapley_vals.values())]
    denom = sum(abs(val) for val in updated)
    if denom == 0:
        return np.array(base), score
    return np.array(updated) / denom, score


def aggregation(
    n_agents: int,
    n_adversarial: int,
    dataset: Iterable[tuple],
    model: str,
    output_file: str,
):
    """Run disconnected aggregation workflow from the original `multi_agents_llm.py` script."""

    judge = JudgeClient(MATH_TEMPLATES.judge)

    adversaries = [OllamaClient(MATH_TEMPLATES.adversarial_system, model) for _ in range(n_adversarial)]
    regular = [OllamaClient(MATH_TEMPLATES.system, model) for _ in range(n_agents - n_adversarial)]
    agents = regular + adversaries

    weights = np.array([1 / n_agents] * n_agents)
    results: List[Dict] = []

    for question, answer in dataset:
        responses = []
        solution = {"problem": question, "answer": answer}

        for idx in range(n_agents):
            prompt = f"Question: {question}"
            response = agents[idx].get_response(prompt)
            solution[f"agent {idx}"] = response
            responses.append(response)

        embeddings, weights_dict, compressed = embedding_collections(responses, weights)
        final_answer, _ = generate_response(compressed, embeddings, weights_dict)

        reward_prompt = (
            f"Question: {question}\n"
            f"Team's Proposed Solution: {final_answer}\n"
            f"Reference Solution: {answer}\n"
        )
        reward = judge.get_score(reward_prompt)
        shapley_vals = calculate_shap(compressed, final_answer, embeddings, weights_dict)
        weights, numeric_reward = update_weights(reward, weights_dict, shapley_vals)

        solution["weights"] = weights.tolist()
        solution["final response"] = final_answer
        solution["shapley"] = list(shapley_vals.values())
        solution["reward"] = numeric_reward
        results.append(solution)

        write_data(file_name=output_file, solutions=results)

    return results
