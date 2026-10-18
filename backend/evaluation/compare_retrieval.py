"""Compare vector retrieval, hybrid RRF, and hybrid RRF + CrossEncoder.

Run from backend/: python -m evaluation.compare_retrieval --top-k 5
The evaluation corpus must be ingested into the database first.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.rag.service import retrieve_context  # noqa: E402
from app.rag.hybrid_search import retrieve_hybrid_context  # noqa: E402
from evaluation.evaluate_retrieval import load_dataset, reciprocal_rank  # noqa: E402


def run_retrieval(question: str, top_k: int, method: str) -> list[dict[str, Any]]:
    if method == "vector_only":
        return retrieve_context(question, top_k)
    if method == "hybrid_rrf":
        return retrieve_hybrid_context(question, top_k, rerank=False)
    if method == "hybrid_rrf_crossencoder":
        return retrieve_hybrid_context(question, top_k, rerank=True)
    raise ValueError(f"Método desconhecido: {method}")


def score(dataset: list[dict[str, Any]], top_k: int, method: str) -> dict[str, Any]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    hits: list[bool] = []
    rows: list[dict[str, Any]] = []
    for item in dataset:
        sources = run_retrieval(item["question"], top_k, method)
        names = [str(source["filename"]) for source in sources]
        expected = set(item["expected_sources"])
        found = expected.intersection(names)
        recall = len(found) / len(expected)
        rr = reciprocal_rank(names, expected)
        recalls.append(recall)
        reciprocal_ranks.append(rr)
        hits.append(bool(found))
        rows.append({
            "id": item["id"], "question": item["question"],
            "expected_sources": sorted(expected), "retrieved_sources": names,
            "recall_at_k": round(recall, 4), "reciprocal_rank": round(rr, 4),
            "hit": bool(found),
        })
    return {
        "metrics": {
            "recall_at_k": round(statistics.mean(recalls), 4),
            "mrr": round(statistics.mean(reciprocal_ranks), 4),
            "hit_rate_at_k": round(sum(hits) / len(hits), 4),
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-crossencoder", action="store_true", help="Não executa o método com CrossEncoder")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k deve ser pelo menos 1")
    try:
        dataset = load_dataset(args.dataset)
        methods = ["vector_only", "hybrid_rrf"]
        if not args.skip_crossencoder:
            methods.append("hybrid_rrf_crossencoder")
        report: dict[str, Any] = {
            "questions": len(dataset), "top_k": args.top_k,
            "dataset": str(args.dataset),
            "expected_sources": sorted({s for item in dataset for s in item["expected_sources"]}),
            "methods": {},
            "notes": [
                "Expected source labels must match filenames ingested into the database.",
                "CrossEncoder adds inference latency and may download model weights on first use.",
                "Metrics evaluate source retrieval, not answer faithfulness or answer correctness.",
            ],
        }
        for method in methods:
            print(f"Avaliando {method} ({len(dataset)} perguntas)...", file=sys.stderr)
            report["methods"][method] = score(dataset, args.top_k, method)
        # Convenience summary for easy comparison and README reporting.
        report["summary"] = {name: value["metrics"] for name, value in report["methods"].items()}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Falha na avaliação: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
