"""Compare vector-only retrieval with hybrid retrieval on the same JSONL dataset."""
import argparse
import json
import statistics
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.rag.service import retrieve_context
from app.rag.hybrid_search import retrieve_hybrid_context
from evaluation.evaluate_retrieval import load_dataset, reciprocal_rank


def score(dataset, top_k, hybrid):
    recalls, reciprocal_ranks, hits = [], [], []
    for item in dataset:
        sources = (retrieve_hybrid_context(item["question"], top_k, rerank=False)
                   if hybrid else retrieve_context(item["question"], top_k))
        names = [str(source["filename"]) for source in sources]
        expected = set(item["expected_sources"])
        found = expected.intersection(names)
        recalls.append(len(found) / len(expected))
        reciprocal_ranks.append(reciprocal_rank(names, expected))
        hits.append(bool(found))
    return {"recall_at_k": round(statistics.mean(recalls), 4),
            "mrr": round(statistics.mean(reciprocal_ranks), 4),
            "hit_rate_at_k": round(sum(hits) / len(hits), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    dataset = load_dataset(args.dataset)
    report = {"questions": len(dataset), "top_k": args.top_k,
              "vector_only": score(dataset, args.top_k, False),
              "hybrid_rrf": score(dataset, args.top_k, True),
              "note": "Cross-encoder is disabled here to isolate hybrid retrieval + RRF."}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
