from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings

try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except Exception as exc:
    raise RuntimeError(
        "Ragas is not installed correctly. Install requirements and retry."
    ) from exc


def load_eval_samples(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON array.")
    if limit is not None:
        payload = payload[:limit]
    return payload


def collect_predictions(backend_url: str, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question", "")).strip()
        ground_truth = str(sample.get("ground_truth", "")).strip()
        document_id = str(sample.get("document_id", "")).strip()
        if not question or not document_id:
            continue

        response = requests.post(
            f"{backend_url.rstrip('/')}/api/ask",
            json={"document_id": document_id, "question": question},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        rows.append(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": result.get("evidence", []) or [""],
                "ground_truth": ground_truth,
            }
        )

    return rows


def run_ragas(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No predictions were collected for evaluation.")

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
            "ground_truth": [row["ground_truth"] for row in rows],
        }
    )

    settings = get_settings()
    if not settings.has_groq or settings.groq_api_key is None:
        raise ValueError("GROQ_API_KEY is required to run Ragas evaluation with LLM metrics.")

    llm = ChatGroq(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        temperature=0,
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    if hasattr(results, "to_pandas"):
        frame = results.to_pandas()
        metrics = frame.mean(numeric_only=True).to_dict()
        return {
            "metrics": {key: float(value) for key, value in metrics.items()},
            "rows": frame.to_dict(orient="records"),
        }

    if isinstance(results, dict):
        return {"metrics": results, "rows": []}

    return {"metrics": {"raw": str(results)}, "rows": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation against backend responses.")
    parser.add_argument(
        "--backend-url",
        default=os.getenv("STREAMLIT_BACKEND_URL", "http://localhost:8000"),
        help="FastAPI backend URL",
    )
    parser.add_argument(
        "--dataset-path",
        default="evaluation/eval_dataset_template.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output-path",
        default="evaluation/ragas_report.json",
        help="Path to output report JSON",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to evaluate",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    samples = load_eval_samples(dataset_path, args.sample_limit)
    rows = collect_predictions(args.backend_url, samples)
    report = run_ragas(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(report["metrics"], ensure_ascii=True, indent=2))
    print(f"Saved detailed report to {output_path}")


if __name__ == "__main__":
    main()
