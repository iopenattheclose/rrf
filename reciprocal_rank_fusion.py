"""
Reciprocal Rank Fusion (RRF) demo: BM25 (lexical) + cosine similarity (semantic).

Install:  pip install rank_bm25 sentence-transformers numpy
Run:      python rrf_demo.py
"""

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DOCUMENTS = [
    # --- Cars: Tesla Model S cluster (0-3) ---
    "The Tesla Model S uses a dual-motor electric drivetrain to reach 60 mph in under two seconds in Plaid trim.",
    "Tesla's Model S Plaid has three motors and over 1,000 horsepower, making it one of the quickest production sedans.",
    "The Model S offers over 400 miles of range, and owners can charge at Tesla Superchargers across the country.",
    "Elon Musk's company launched its flagship luxury electric sedan in 2012, changing how people viewed EVs.",
    # --- Cars: Porsche 911 cluster (4-7) ---
    "The Porsche 911 has kept its rear-engine sports car layout for over sixty years.",
    "Porsche's 911 Carrera uses a flat-six engine mounted behind the rear axle, giving it distinctive handling.",
    "The 911 GT3 is a track-focused Porsche with a high-revving naturally aspirated engine and a manual gearbox option.",
    "Stuttgart's iconic sports coupe, with its round headlights and sloping roofline, is instantly recognisable.",
    # --- Cars: EV charging and batteries cluster (8-11) ---
    "Solid-state batteries promise longer range and faster charging for future electric vehicles.",
    "Electric SUVs like the Hyundai Ioniq 5 support 800-volt ultra-fast charging, adding 200 miles in about 20 minutes.",
    "Lithium-ion batteries power phones, laptops and electric vehicles thanks to their high energy density.",
    "Public fast-charging stations let drivers top up an EV battery in under half an hour on long trips.",
    # --- Cars: self-driving cluster (12-14) ---
    "Self-driving cars use cameras, radar and lidar to detect lanes, pedestrians and other vehicles.",
    "Waymo's robotaxis operate without a human driver in several US cities, relying on lidar and detailed maps.",
    "Autonomous vehicles must handle rare edge cases like construction zones and unpredictable pedestrians.",
    # --- Cars: Toyota / hybrid cluster (15-16) ---
    "The Toyota Corolla is one of the best-selling cars ever, known for reliability and low maintenance costs.",
    "Toyota's Prius popularised hybrid cars by pairing a petrol engine with an electric motor to save fuel.",
    # --- Same keyword, different meaning: Jaguar (17-18) ---
    "The Jaguar F-Type is a British sports car with a supercharged V8 engine and a sleek aluminium body.",
    "The jaguar is a big cat native to the Americas, known for its powerful bite and love of water.",
    # --- Space: James Webb cluster (19-21) ---
    "The James Webb Space Telescope observes the universe in infrared, revealing some of the earliest galaxies.",
    "Webb's infrared images of distant galaxies show light that has travelled for over thirteen billion years.",
    "NASA's largest space observatory sits at the Sun-Earth L2 point, shielded from heat by a five-layer sunshield.",
    # --- Space: Moon missions cluster (22-25) ---
    "NASA's Artemis program aims to return astronauts to the lunar surface and build a long-term presence there.",
    "ISRO's Chandrayaan-3 mission made India the first country to land near the Moon's south pole.",
    "Artemis II will send a crew around the Moon, testing the Orion spacecraft before a future landing.",
    "Chandrayaan-3's Vikram lander and Pragyan rover found evidence of sulphur on the lunar surface.",
    # --- Space: Starship / rockets cluster (26-28) ---
    "SpaceX Starship is a fully reusable rocket designed to carry crew and cargo to the Moon and Mars.",
    "Starship's Super Heavy booster can be caught by the launch tower's mechanical arms for rapid reuse.",
    "A rocket works by Newton's third law: expelling exhaust downward pushes the vehicle upward.",
    # --- Space: Mars cluster (29-30) ---
    "Mars rovers such as Perseverance search for signs of ancient microbial life and collect rock samples.",
    "The Red Planet's thin atmosphere and cold surface make human settlement extremely challenging.",
    # --- Space: stellar objects (31-32) ---
    "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape.",
    "Neutron stars are the dense collapsed cores of massive stars, spinning up to hundreds of times per second.",
    # --- AI: RAG / embeddings / search cluster (33-37) ---
    "Retrieval-augmented generation combines a search step with an LLM so answers are grounded in documents.",
    "RAG systems fetch relevant passages from a knowledge base and feed them to a language model as context.",
    "Vector databases store embeddings and use approximate nearest neighbour search to find similar items fast.",
    "Embeddings map words or sentences to dense vectors so that semantically similar text lies close together.",
    "BM25 is a keyword-based ranking function that scores documents by term frequency and document length.",
    # --- AI: LLM / transformer cluster (38-40) ---
    "Large language models are trained on huge text datasets to predict the next token and generate fluent text.",
    "Transformers use self-attention to weigh the relevance of every token to every other token in a sequence.",
    "ChatGPT-style chatbots are built on transformer models fine-tuned with human feedback.",
    # --- AI: other (41-42) ---
    "Diffusion models generate images by gradually removing noise from random pixels guided by a text prompt.",
    "Overfitting happens when a model memorises training data and performs poorly on unseen examples.",
    # --- Tech advancements: quantum cluster (43-44) ---
    "Quantum computers use qubits that can be in superposition, promising speedups for certain problems.",
    "Qubits are fragile, so quantum computers need error correction and extremely cold operating temperatures.",
    # --- Tech advancements: chips / networks / other (45-46) ---
    "5G networks offer higher bandwidth and lower latency than 4G, enabling real-time applications.",
    "The Apple M-series chips use ARM architecture to deliver strong performance with low power consumption.",
    # --- Same keyword, different meaning: Apple (47) ---
    "Apple orchards harvest fruit in autumn, and varieties like Fuji and Gala are popular for eating fresh.",
    # --- Tech advancements: biotech / energy (48-49) ---
    "CRISPR gene editing lets scientists cut and modify DNA with high precision.",
    "Solar panel efficiency has improved steadily, making renewable electricity cheaper than coal in many regions.",
]


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens."""
    return re.findall(r"\w+", text.lower())


# ----------------------------------------------------------------------
# Retrievers: each returns a full ranking as a list of (doc_id, score)
# ----------------------------------------------------------------------
class BM25Retriever:
    """Lexical search using BM25."""

    name = "BM25"

    def __init__(self, documents: list[str]):
        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def rank(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order]


class DenseRetriever:
    """Semantic search: sentence embeddings + cosine similarity."""

    name = "Cosine"

    def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # Normalised vectors -> dot product equals cosine similarity
        self.doc_vectors = self.model.encode(documents, normalize_embeddings=True)

    def rank(self, query: str) -> list[tuple[int, float]]:
        q_vec = self.model.encode(query, normalize_embeddings=True)
        scores = self.doc_vectors @ q_vec
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order]


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------
class RRFFuser:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, rankings: list[list[int]], k: int | None = None) -> list[tuple[int, float]]:
        """rankings: list of ranked doc_id lists (best first). Returns [(doc_id, rrf_score)]."""
        k = self.k if k is None else k
        scores = defaultdict(float)
        for ranked_ids in rankings:
            for rank, doc_id in enumerate(ranked_ids, start=1):  # rank starts at 1
                scores[doc_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])


# ----------------------------------------------------------------------
# Orchestrator (this is what a Flask route will call later)
# ----------------------------------------------------------------------
class HybridSearch:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)
        self.fuser = RRFFuser()

    def search(self, query: str, k: int = 60, top_n: int = 10) -> dict:
        """Returns {"bm25": [...], "cosine": [...], "rrf": [...]}, each with the top_n rows."""
        bm25_ranking = self.bm25.rank(query)
        dense_ranking = self.dense.rank(query)

        # doc_id -> rank (1-based), used to show each doc's rank in the RRF table
        bm25_rank = {doc_id: r for r, (doc_id, _) in enumerate(bm25_ranking, start=1)}
        dense_rank = {doc_id: r for r, (doc_id, _) in enumerate(dense_ranking, start=1)}

        fused = self.fuser.fuse(
            [[d for d, _ in bm25_ranking], [d for d, _ in dense_ranking]], k=k
        )

        bm25_rows = [
            {"rank": r, "doc_id": d, "score": s, "text": self.documents[d]}
            for r, (d, s) in enumerate(bm25_ranking[:top_n], start=1)
        ]
        cosine_rows = [
            {"rank": r, "doc_id": d, "score": s, "text": self.documents[d]}
            for r, (d, s) in enumerate(dense_ranking[:top_n], start=1)
        ]
        rrf_rows = [
            {
                "rank": r,
                "doc_id": d,
                "score": s,
                "bm25_rank": bm25_rank[d],
                "cosine_rank": dense_rank[d],
                "text": self.documents[d],
            }
            for r, (d, s) in enumerate(fused[:top_n], start=1)
        ]
        return {"bm25": bm25_rows, "cosine": cosine_rows, "rrf": rrf_rows}


# ----------------------------------------------------------------------
# Console output
# ----------------------------------------------------------------------
def short(text: str, width: int = 70) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_results(query: str, k: int, results: dict) -> None:
    print(f"\nQuery: {query!r}   (k = {k})")

    # Table 1: BM25
    print("\n=== Table 1: BM25 (keyword) ===")
    header = f"{'Rank':<6}{'Doc':<5}{'BM25 score':>11}  Text"
    print(header)
    print("-" * (len(header) + 60))
    for r in results["bm25"]:
        print(f"{r['rank']:<6}{r['doc_id']:<5}{r['score']:>11.3f}  {short(r['text'])}")

    # Table 2: Semantic (cosine)
    print("\n=== Table 2: Semantic (cosine similarity) ===")
    header = f"{'Rank':<6}{'Doc':<5}{'Cosine':>11}  Text"
    print(header)
    print("-" * (len(header) + 60))
    for r in results["cosine"]:
        print(f"{r['rank']:<6}{r['doc_id']:<5}{r['score']:>11.3f}  {short(r['text'])}")

    # Table 3: RRF (with each doc's rank in the other two tables)
    print(f"\n=== Table 3: RRF fusion (k = {k}) ===")
    header = f"{'Rank':<6}{'Doc':<5}{'BM25 rank':<10}{'Cos rank':<9}{'RRF score':>10}  Text"
    print(header)
    print("-" * (len(header) + 60))
    for r in results["rrf"]:
        print(
            f"{r['rank']:<6}{r['doc_id']:<5}{r['bm25_rank']:<10}{r['cosine_rank']:<9}"
            f"{r['score']:>10.5f}  {short(r['text'])}"
        )


def main() -> None:
    print(f"Loading {len(DOCUMENTS)} documents and embedding model...")
    engine = HybridSearch(DOCUMENTS)

    while True:
        query = input("\nEnter a query (or press Enter to quit): ").strip()
        if not query:
            break
        k_text = input("k value [60]: ").strip()
        k = int(k_text) if k_text.isdigit() else 60
        print_results(query, k, engine.search(query, k=k, top_n=10))


if __name__ == "__main__":
    main()