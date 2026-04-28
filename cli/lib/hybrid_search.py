import json
import os
from time import sleep
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    format_search_result,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        raw_results = self.idx.bm25_search(query, limit)
        results = []
        for doc, score in raw_results:
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return results

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1 / (k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[dict], semantic_results: list[dict], k: int = RRF_K
) -> list[dict]:
    rrf_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    sorted_items = sorted(
        rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
    )

    rrf_results = []
    for doc_id, data in sorted_items:
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return rrf_results


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance: str | None = None,
    rerank: str | None = None,
    evaluate: int = 0
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = query

    if enhance == "spell":
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        )
        enhanced_query = response.text
    elif enhance == "rewrite":
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        )
        enhanced_query = response.text
    elif enhance == "expand":
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
        )
        enhanced_query = response.text

    search_limit = limit

    if rerank:
        search_limit *= 5

    results = searcher.rrf_search(enhanced_query, k, search_limit)

    if rerank == "cross_encoder":
        rerank_scores = {}
        # Force CPU to avoid CUDA compatibility failures on older GPUs.
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
        pairs = []
        for doc in results:
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        scores = cross_encoder.predict(pairs)

        for i, doc in enumerate(results):
            rerank_scores[doc['id']] = scores[i]
        
        results = sorted(results, key=lambda x: rerank_scores[x['id']], reverse=True)
        
    elif rerank == "batch":
        rerank_scores = {}
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{json.dumps(results)}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:""")

        ids = json.loads(response.text)
        for i, id in enumerate(ids):
            rerank_scores[int(id)] = 100 - i
        
        results = sorted(results, key=lambda x: rerank_scores.get(int(x.get('id') or 0) or 0) or 0, reverse=True)

    elif rerank == "individual":
        rerank_scores = {}

        for doc in results:
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:""")
            rerank_scores[doc['id']] = float(response.text)
            sleep(3)

            if rerank_scores:
                results = sorted(results, key=lambda x: rerank_scores[x['id']], reverse=True)

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query if enhance else None,
        "query": query,
        "k": k,
        "results": results,
    }

def evaluate_search_results(query: str, results: list[dict]) -> list[int]:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join([f"{result['title']} - {result['document']}" for result in results])}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]""",
    )
    raw_scores = response.text if getattr(response, "text", None) else "[]"
    try:
        parsed_scores = json.loads(raw_scores)
    except (TypeError, json.JSONDecodeError):
        parsed_scores = []

    if not isinstance(parsed_scores, list):
        parsed_scores = []

    scores: list[int] = []
    for score in parsed_scores[: len(results)]:
        try:
            normalized = int(score)
        except (TypeError, ValueError):
            normalized = 0
        scores.append(max(0, min(3, normalized)))

    if len(scores) < len(results):
        scores.extend([0] * (len(results) - len(scores)))

    return scores