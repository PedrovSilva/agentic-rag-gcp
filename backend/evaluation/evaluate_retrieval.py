"""Evaluate retrieval quality against a versioned JSONL dataset.

Run from the backend directory: python -m evaluation.evaluate_retrieval --top-k 5
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from app.rag.service import retrieve_context  # noqa: E402

DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido na linha {line_number}: {exc}") from exc
            required = {"id", "question", "expected_answer", "expected_sources"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Linha {line_number} sem campos: {', '.join(sorted(missing))}")
            if not isinstance(record["expected_sources"], list) or not record["expected_sources"]:
                raise ValueError(f"Linha {line_number}: expected_sources deve ser uma lista não vazia")
            records.append(record)
    if not records:
        raise ValueError(f"Nenhum caso de avaliação encontrado em {path}")
    return records


def reciprocal_rank(retrieved: list[str], expected: set[str]) -> float:
    for rank, name in enumerate(retrieved, start=1):
        if name in expected:
            return 1.0 / rank
    return 0.0


def evaluate(dataset: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("--top-k deve ser pelo menos 1")
    recalls, reciprocal_ranks, rows = [], [], []
    for item in dataset:
        results = retrieve_context(item["question"], top_k=top_k)
        retrieved = [str(source["filename"]) for source in results]
        expected = set(item["expected_sources"])
        hits = expected.intersection(retrieved)
        recall = len(hits) / len(expected)
        mrr = reciprocal_rank(retrieved, expected)
        recalls.append(recall)
        reciprocal_ranks.append(mrr)
        rows.append({"id": item["id"], "question": item["question"],
                     "expected_sources": sorted(expected), "retrieved_sources": retrieved,
                     "recall_at_k": round(recall, 4), "reciprocal_rank": round(mrr, 4),
                     "hit": bool(hits)})
    return {"dataset": "dataset.jsonl", "total_questions": len(dataset), "k": top_k,
            "metrics": {"recall_at_k": round(statistics.mean(recalls), 4),
                        "mrr": round(statistics.mean(reciprocal_ranks), 4),
                        "hit_rate_at_k": round(sum(r["hit"] for r in rows) / len(rows), 4)},
            "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Caminho opcional para relatório JSON")
    args = parser.parse_args()
    try:
        report = evaluate(load_dataset(args.dataset), args.top_k)
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
