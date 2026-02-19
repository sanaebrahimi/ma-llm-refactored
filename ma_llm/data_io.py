"""Dataset readers and JSON result writers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def read_data_qa(file_path: str = "./data/ResearchQA/subq_QA_4.json") -> Tuple[List[str], List[List[str]]]:
    """Read QA dataset with question and list of subquestions."""

    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    questions = [item["question"] for item in data]
    subquestions = [item["subquestions"] for item in data]
    return questions, subquestions


def read_data_mathematics(file_path: str = "./data/algebra__linear_2d.txt") -> Tuple[List[str], List[str]]:
    """Read alternating-line text file of math questions and answers."""

    questions: List[str] = []
    solutions: List[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle.readlines()):
            if index % 2 == 0:
                questions.append(line.rstrip())
            else:
                solutions.append(line.rstrip())
    return questions, solutions


def read_math_json(file_path: str) -> Tuple[List[str], List[str]]:
    """Read JSON list with ``question/problem`` and ``solution`` keys."""

    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    questions: List[str] = []
    solutions: List[str] = []
    for item in data:
        prompt = item.get("question", item.get("problem", ""))
        questions.append(prompt)
        solutions.append(item.get("solution", ""))
    return questions, solutions


def read_gsm8k(file_path: str) -> Tuple[List[str], List[str]]:
    """Read GSM8K JSONL file."""

    questions: List[str] = []
    solutions: List[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            questions.append(item["question"])
            solutions.append(item["answer"])
    return questions, solutions


def read_mmlu_mathematics(file_path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Read MMLU-style CSV with question, four choices, and solution letter."""

    frame = pd.read_csv(file_path)
    questions = frame.iloc[:, 0].tolist()
    choices = frame.iloc[:, 1:-1].values.tolist()
    solutions = frame.iloc[:, -1].tolist()
    return questions, choices, solutions


def read_data_code(file_path: str = "./data/HumanEval.jsonl") -> Tuple[List[str], List[str], List[str], List[str]]:
    """Read HumanEval-like JSONL with prompt, canonical solution, test and task id."""

    questions: List[str] = []
    solutions: List[str] = []
    tests: List[str] = []
    task_ids: List[str] = []

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            questions.append(item["prompt"])
            solutions.append(item["canonical_solution"])
            tests.append(item["test"])
            task_ids.append(item["task_id"])

    return questions, solutions, tests, task_ids


def generate_choice_map(file_path: str) -> Dict[str, Dict[str, str]]:
    """Create map from question to answer-choice text by letter."""

    questions, choices, _ = read_mmlu_mathematics(file_path)
    return {
        question: {"A": c[0], "B": c[1], "C": c[2], "D": c[3]}
        for question, c in zip(questions, choices)
    }


def get_question_options(question_text: str, file_path: str) -> List[str]:
    """Return answer options for one question from an MMLU CSV."""

    frame = pd.read_csv(file_path)
    question_column = frame.columns[0]
    row = frame[frame[question_column] == question_text]
    if row.empty:
        return []
    return row.iloc[0, 1:-1].tolist()


def write_data(folder_path: str = "./", file_name: str = "resultsX.json", solutions: Any = None) -> Path:
    """Write result payload to disk and return the output path."""

    payload = {} if solutions is None else solutions
    path = Path(os.path.join(folder_path, file_name))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    return path
