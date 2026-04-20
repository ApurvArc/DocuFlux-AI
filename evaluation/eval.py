"""
evaluation/eval.py

Evaluation framework for the RAG system.

- Retrieval evaluation: MRR, nDCG, keyword coverage (fully local, no API needed)
- Answer evaluation: LLM-as-judge (OPTIONAL - requires a local LM Studio server
  or an API key configured in .env)
"""

import sys
import math
import os
import json
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from evaluation.test import TestQuestion, load_tests
from core.answer import answer_question, fetch_context
from core.config import LM_STUDIO_BASE, MODEL_PROVIDERS

_EVAL_MODEL = MODEL_PROVIDERS["Local (LM Studio)"]["model"]


load_dotenv(override=True)

_AI_EVAL_OVERRIDE = os.getenv("AI_EVAL_ENABLED", "").lower()
AI_EVAL_ENABLED: bool = _AI_EVAL_OVERRIDE != "false"
_judge_client = None


def _get_judge_client():
    """Lazily initialise LM Studio client, return None if unavailable."""
    global _judge_client
    if _judge_client is not None:
        return _judge_client
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LM_STUDIO_BASE, api_key="lm-studio")
        client.models.list()
        _judge_client = client
        return client
    except Exception:
        return None


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""
    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total keywords to find")
    keyword_coverage: float = Field(description="% of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""
    feedback: str = Field(description="Brief feedback on answer quality")
    accuracy: float = Field(description="Factual correctness vs reference (1-5)")
    completeness: float = Field(description="Coverage of all aspects (1-5)")
    relevance: float = Field(description="Relevance to question, no extra info (1-5)")


def _coerce_answer_eval(raw: str) -> AnswerEval:
    """Accept slightly malformed judge JSON instead of failing hard."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Judge response must be a JSON object")

    feedback = data.get("feedback", "")
    if not isinstance(feedback, str):
        feedback = str(feedback)

    return AnswerEval(
        feedback=feedback or "No written feedback provided.",
        accuracy=float(data["accuracy"]),
        completeness=float(data["completeness"]),
        relevance=float(data["relevance"]),
    )


def safe_console_text(value: object) -> str:
    """Render text safely on cp1252 Windows consoles."""
    return str(value).encode("cp1252", errors="replace").decode("cp1252")


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Reciprocal rank for a single keyword (case-insensitive)."""
    kw = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if kw in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    kw = keyword.lower()
    relevances = [1 if kw in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
    dcg = calculate_dcg(relevances, k)
    idcg = calculate_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """Evaluate retrieval quality for one test question (no API needed)."""
    retrieved_docs = fetch_context(test.question)

    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    keywords_found = sum(1 for s in mrr_scores if s > 0)

    return RetrievalEval(
        mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        ndcg=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        keywords_found=keywords_found,
        total_keywords=len(test.keywords),
        keyword_coverage=(keywords_found / len(test.keywords) * 100) if test.keywords else 0.0,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval | None, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge.
    Returns (None, generated_answer, docs) if AI eval is unavailable.
    """
    generated_answer, retrieved_docs = answer_question(test.question)

    if not AI_EVAL_ENABLED:
        return None, generated_answer, retrieved_docs

    client = _get_judge_client()
    if client is None:
        return None, generated_answer, retrieved_docs

    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator assessing RAG answer quality. "
                "Compare the generated answer to the reference answer. "
                "Respond in JSON with keys: feedback, accuracy, completeness, relevance (each 1-5)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {test.question}\n\n"
                f"Generated Answer: {generated_answer}\n\n"
                f"Reference Answer: {test.reference_answer}\n\n"
                "Evaluate on accuracy, completeness, and relevance (1-5 each). "
                "Respond ONLY with JSON."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=_EVAL_MODEL,
            messages=judge_messages,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        answer_eval = _coerce_answer_eval(raw)
        return answer_eval, generated_answer, retrieved_docs
    except Exception as e:
        print(f"[Answer eval skipped] {e}")
        return None, generated_answer, retrieved_docs


def evaluate_all_retrieval():
    """Yield (test, RetrievalEval, progress) for each test."""
    tests = load_tests()
    total = len(tests)
    for i, test in enumerate(tests):
        result = evaluate_retrieval(test)
        yield test, result, (i + 1) / total


def evaluate_all_answers():
    """
    Yield (test, AnswerEval | None, progress) for each test.
    AnswerEval is None when AI evaluation is unavailable.
    """
    tests = load_tests()
    total = len(tests)
    for i, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        yield test, result, (i + 1) / total


def run_cli_evaluation(test_number: int):
    tests = load_tests()
    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]
    print(f"\n{'=' * 80}\nTest #{test_number}\n{'=' * 80}")
    print(f"Question : {test.question}")
    print(f"Keywords : {test.keywords}")
    print(f"Category : {test.category}")
    print(f"Reference: {test.reference_answer}")

    print(f"\n{'=' * 80}\nRetrieval Evaluation\n{'=' * 80}")
    r = evaluate_retrieval(test)
    print(f"MRR               : {r.mrr:.4f}")
    print(f"nDCG              : {r.ndcg:.4f}")
    print(f"Keywords found    : {r.keywords_found}/{r.total_keywords}")
    print(f"Keyword coverage  : {r.keyword_coverage:.1f}%")

    print(f"\n{'=' * 80}\nAnswer Evaluation\n{'=' * 80}")
    eval_result, generated_answer, _ = evaluate_answer(test)
    print(f"\nGenerated Answer:\n{safe_console_text(generated_answer)}")
    if eval_result is None:
        print("\n[Warning] AI evaluation skipped (LM Studio not running or AI_EVAL_ENABLED=false)")
    else:
        print(f"\nFeedback   : {safe_console_text(eval_result.feedback)}")
        print(f"Accuracy   : {eval_result.accuracy:.2f}/5")
        print(f"Completeness: {eval_result.completeness:.2f}/5")
        print(f"Relevance  : {eval_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    if len(sys.argv) != 2:
        print("Hint: You can pass a specific test number (e.g. python evaluation/eval.py 2)")
        print("No test number provided to script. Defaulting to testing row #0...\n")
        run_cli_evaluation(0)
    else:
        try:
            run_cli_evaluation(int(sys.argv[1]))
        except ValueError:
            print("Error: test_row_number must be an integer")
            sys.exit(1)


if __name__ == "__main__":
    main()
