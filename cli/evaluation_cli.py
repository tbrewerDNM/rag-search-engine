import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    search_instance = HybridSearch(load_movies())    

    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    test_cases = golden_dataset.get("test_cases", [])

    print(f"k={limit}\n\n")
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        results = search_instance.rrf_search(query, 60, limit)

        retrieved_titles = [result["title"] for result in results]
        hits = sum(1 for title in retrieved_titles if title in relevant_docs)

        precision = hits / limit if limit > 0 else 0.0
        recall = hits / len(relevant_docs) if relevant_docs else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print(f"""- Query: {query}
  - Precision@{limit}: {precision:.4f}
  - Recall@{limit}: {recall:.4f}
  - F1 Score: {f1_score:.4f}
  - Retrieved: {", ".join(retrieved_titles)}
  - Relevant: {", ".join(relevant_docs)}
        """)
        print()


if __name__ == "__main__":
    main()