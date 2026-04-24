#!/usr/bin/env python3

import argparse
from lib.semantic_search import embed_query_text, embed_text, perform_search, verify_embeddings, verify_model

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()