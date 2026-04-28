import argparse

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summ_parser = subparsers.add_parser(
        "summarize", help="Summarize a document"
    )
    summ_parser.add_argument("query", type=str, help="Document to summarize")
    summ_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    cit_parser = subparsers.add_parser(
        "citations", help="Summarize a document"
    )
    cit_parser.add_argument("query", type=str, help="Document to summarize")
    cit_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    q_parser = subparsers.add_parser(
        "question", help="Summarize a document"
    )
    q_parser.add_argument("query", type=str, help="Document to summarize")
    q_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            hs = HybridSearch(load_movies())
            docs = hs.rrf_search(query, 60, 5)
            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {docs}

            Answer:"""
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt
            )
            
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("RAG Response:")
            print(response.text)
            print()
        case "summarize":
            query = args.query
            hs = HybridSearch(load_movies())
            docs = hs.rrf_search(query, 60, args.limit)
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt
            )
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("LLM Summary:")
            print(response.text)
            print()
        case "citations":
            query = args.query
            hs = HybridSearch(load_movies())
            docs = hs.rrf_search(query, 60, args.limit)
            prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of Hoopla, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

            Query: {query}

            Documents:
            {docs}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt
            )
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("LLM Answer:")
            print(response.text)
            print()
        case "question":
            query = args.query
            hs = HybridSearch(load_movies())
            docs = hs.rrf_search(query, 60, args.limit)
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

            Question: {query}

            Documents:
            {docs}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt
            )
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("Answer:")
            print(response.text)
            print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()