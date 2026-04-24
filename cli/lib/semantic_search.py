from lib.search_utils import cosine_similarity, load_movies
from sentence_transformers import SentenceTransformer
import numpy as np
import os


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embeddings = []
        self.documents = []
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("empty text")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents

        embeddings = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            embeddings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(embeddings, show_progress_bar=True)

        with open("cache/movie_embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        embeddings = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            embeddings.append(f"{doc['title']}: {doc['description']}")

        if os.path.exists("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)

        if len(self.embeddings) != len(embeddings):
            return self.build_embeddings(documents)

        return self.embeddings

    def search(self, query, limit):
        if not len(self.embeddings):
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        embedding = self.generate_embedding(query)

        scores = []
        for i in range(len(self.documents)):
            cos_sim = cosine_similarity(
                embedding,
                self.embeddings[i]
            )

            scores.append(
                (cos_sim, self.documents[i])
            )

        scores = sorted(scores, key=lambda x: x[0], reverse=True)

        if len(scores) > limit:
            scores = scores[0:limit]

        return [{
            'score': score[0],
            'title': score[1]['title'],
            'description': score[1]['description']
        } for score in scores]


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(ss.documents)}")
    print(f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def perform_search(query, limit):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})\n  {result['description']}")
