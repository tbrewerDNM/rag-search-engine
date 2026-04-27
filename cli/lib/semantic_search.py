from lib.search_utils import cosine_similarity, load_movies
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import json


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


def chunk_text(text: str, chunk_size=200, overlap=0) -> list[str]:
    tokens = text.split(" ")

    chunks = []
    current_tokens = []

    i = 0
    while i < len(tokens):
        current_tokens.append(tokens[i])

        if len(current_tokens) >= chunk_size:
            chunks.append(current_tokens)
            current_tokens = []
            i -= overlap
        
        i += 1
    
    if len(current_tokens) > 0:
        chunks.append(current_tokens)

    return chunks


def semantic_chunk_text(text: str, chunk_size=200, overlap=0) -> list[list[str]]:
    tokens = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_tokens = []

    i = 0
    while i < len(tokens):
        current_tokens.append(tokens[i])

        if len(current_tokens) >= chunk_size:
            chunks.append(current_tokens)
            current_tokens = []
            i -= overlap
        
        i += 1
    
    if len(current_tokens) > 0:
        chunks.append(current_tokens)

    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        
        all_chunks = []
        self.chunk_metadata = []

        for idx, doc in enumerate(documents):
            self.document_map[doc['id']] = doc

            if not doc['description']:
                continue
            
            chunks = semantic_chunk_text(doc['description'],  4, 1)

            all_chunks.extend(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                self.chunk_metadata.append({
                    'movie_idx': idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks), 
                })
        
        self.chunk_embeddings = self.model.encode(all_chunks)

        with open("cache/chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)
        
        with open("cache/chunk_metadata.json", "w") as f:
            f.write(json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2))
        
        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc['id']] = doc
        
        chunk_embeddings_exists = os.path.exists("cache/chunk_embeddings.npy")
        chunk_json_exists = os.path.exists("cache/chunk_metadata.json")

        if chunk_embeddings_exists and chunk_json_exists:
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
        
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings
    
        return self.build_chunk_embeddings(documents)
