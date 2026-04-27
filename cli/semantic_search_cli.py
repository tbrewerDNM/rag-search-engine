#!/usr/bin/env python3

import argparse
from lib.search_utils import load_movies
from lib.semantic_search import ChunkedSemanticSearch, chunk_text, embed_query_text, embed_text, perform_search, semantic_chunk_text, verify_embeddings, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify model")

    et_parser = subparsers.add_parser("embed_text", help="Verify model")
    et_parser.add_argument("text", type=str, help="Term to get BM25 TF score for")

    subparsers.add_parser("verify_embeddings", help="Verify model")

    eq_parser = subparsers.add_parser("embedquery", help="Verify model")
    eq_parser.add_argument("query", type=str, help="Term to get BM25 TF score for")

    search_parser = subparsers.add_parser("search", help="Verify model")
    search_parser.add_argument("query", type=str, help="Term to get BM25 TF score for")
    search_parser.add_argument("--limit", type=int, default=5, nargs='?', required=False, help="Term to get BM25 TF score for")

    chunk_parser = subparsers.add_parser("chunk", help="Verify model")
    chunk_parser.add_argument("text", type=str, help="Term to get BM25 TF score for")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, nargs='?', required=False, help="Term to get BM25 TF score for")
    chunk_parser.add_argument("--overlap", type=int, default=0, nargs='?', required=False, help="Term to get BM25 TF score for")

    semchunk_parser = subparsers.add_parser("semantic_chunk", help="Verify model")
    semchunk_parser.add_argument("text", type=str, help="Term to get BM25 TF score for")
    semchunk_parser.add_argument("--max-chunk-size", type=int, default=4, nargs='?', required=False, help="Term to get BM25 TF score for")
    semchunk_parser.add_argument("--overlap", type=int, default=0, nargs='?', required=False, help="Term to get BM25 TF score for")

    subparsers.add_parser("embed_chunks", help="Verify model")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            perform_search(args.query, args.limit)
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {' '.join(chunk)}")
        case "semantic_chunk":
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {' '.join(chunk)}")
        case "embed_chunks":
            movies = load_movies()
            css = ChunkedSemanticSearch()
            css.load_or_create_embeddings(movies)
            print(f"Generated {len(css.chunk_embeddings)} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()