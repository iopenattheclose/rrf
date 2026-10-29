# rrf
Implementation of reciprocal rank fusion

# Reciprocal Rank Fusion (RRF) Demo

A small, single-script demo of hybrid search. It runs two retrievers over a hardcoded corpus of 50 documents, then merges their rankings with **Reciprocal Rank Fusion**.

- **BM25** - lexical (keyword) search
- **Cosine similarity** - semantic search using sentence embeddings
- **RRF** - rank-based fusion of the two

The console output shows each document's rank under all three methods, so you can see how fusion changes the order.

## What is RRF?

Reciprocal Rank Fusion merges several ranked result lists into one. It uses only each document's **rank position** in each list, never the raw scores.

```
RRF(d) = sum over lists of  1 / (k + rank_i(d))
```

- `rank_i(d)` is the document's position (1, 2, 3, ...) in list *i*
- `k` is a smoothing constant, 60 by default
- A document missing from a list contributes nothing for that list

Documents are then sorted by RRF score, highest first.

### Why rank-based?

BM25 scores are unbounded (0 to 20+), while cosine similarity lies in [-1, 1]. Adding or averaging them directly needs fragile normalization. Ranks are comparable across any scoring system, so RRF avoids the problem.

### Worked example (k = 60)

| Doc | BM25 rank | Cosine rank | RRF score |
|-----|-----------|-------------|-----------|
| A | 1 | 3 | 1/61 + 1/63 = 0.03226 |
| B | 2 | 1 | 1/62 + 1/61 = 0.03253 |
| C | 3 | 2 | 1/63 + 1/62 = 0.03200 |

Final order: B, A, C. B wins because it ranks highly in both lists, even though it was not #1 in BM25.

### Effect of k

A larger `k` flattens the difference between top ranks, so agreement across lists matters more than being #1 in one. A smaller `k` rewards top-ranked hits more heavily.

## Setup

Requires Python 3.10 or newer (the script uses `int | None` type hints).

```bash
python -m venv venv
source venv/bin/activate
python -m pip install rank-bm25 sentence-transformers numpy
```

Install the packages with the same interpreter that runs the script (`python -m pip`), otherwise you can get `ModuleNotFoundError: No module named 'rank_bm25'`.

The first run downloads the `all-MiniLM-L6-v2` embedding model.

## Usage

```bash
python rrf_demo.py
```

The script prompts for a query and a `k` value (press Enter for 60), then prints the top 10 results:

```
RRF#  Doc  BM25 rank Cos rank   BM25  Cosine      RRF  Text
```

| Column | Meaning |
|--------|---------|
| RRF# | Final rank after fusion |
| Doc | Document ID (its index in `DOCUMENTS`) |
| BM25 rank / Cos rank | The document's rank under each individual method |
| BM25 / Cosine | Raw scores from each method |
| RRF | Fused score |

An empty query exits.

## The corpus

`DOCUMENTS` holds 50 short documents on cars, space, AI and tech advancements. They are built in **clusters of related documents**, so several docs compete for the same query:

- Tesla Model S (4), Porsche 911 (4), EV charging and batteries (4), self-driving (3), Toyota/hybrid (2)
- James Webb (3), Moon missions (4), Starship (3), Mars (2)
- RAG/embeddings/search (5), LLMs/transformers (3), quantum computing (2)

Within a cluster, some docs share keywords with the query and some are paraphrases with different words. Two keyword traps reuse a word with a different meaning: **Jaguar** (car vs animal) and **Apple** (chips vs fruit).

A document's ID is its index in the list (0-49, in cluster order).

### Queries to try

Run each with k=60 and k=1 and compare.

| Query | What it shows |
|-------|---------------|
| `Tesla Model S` | BM25 favours docs saying "Tesla"; only cosine finds the doc that never names the car |
| `Porsche 911` | Same pattern with the "Stuttgart's iconic sports coupe" doc |
| `jaguar` | BM25 sees car and animal as equal matches; embeddings may prefer one |
| `jaguar sports car` vs `jaguar big cat` | Embeddings separate the two senses |
| `apple chip` vs `apple fruit` | Same idea with Apple |
| `Moon landing` | Fusion pools Artemis and Chandrayaan docs from both lists |
| `how do chatbots find relevant documents` | Few shared keywords with the RAG docs, so cosine dominates |

## How the code works

### `tokenize(text)`

Lowercases the text and extracts runs of letters and digits with a regex. `"Tesla's Model S"` becomes `["tesla", "s", "model", "s"]`. Only BM25 uses it; the embedding model tokenizes on its own.

### `BM25Retriever`

- `__init__` tokenizes all documents and builds a `BM25Okapi` index (term frequencies, document lengths, word rarity). Runs once.
- `rank(query)` gets one BM25 score per document, sorts descending with `np.argsort(-scores, kind="stable")` (stable, so ties keep corpus order), and returns `(doc_id, score)` pairs, best first.

### `DenseRetriever`

- `__init__` loads the embedding model and encodes all documents into a 50 x 384 matrix with `normalize_embeddings=True`. With unit-length vectors, **the dot product equals cosine similarity**, so no separate cosine function is needed.
- `rank(query)` embeds the query the same way, computes all 50 scores with `doc_vectors @ q_vec`, sorts, and returns `(doc_id, score)` pairs.

### `RRFFuser`

```python
def fuse(self, rankings, k=None):
    k = self.k if k is None else k
    scores = defaultdict(float)
    for ranked_ids in rankings:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

- `rankings` is a list of ranked doc-ID lists, one per method.
- `enumerate(..., start=1)` gives the first doc rank 1.
- `scores[doc_id] += 1/(k + rank)` is the RRF formula; a doc in both lists receives two contributions.
- Returns `(doc_id, rrf_score)` pairs, highest first.

### `HybridSearch`

The orchestrator. `__init__` builds both retrievers and the fuser once (the slow part is loading the model and embedding the docs).

`search(query, k=60, top_n=10)`:

1. Gets the full BM25 and cosine rankings.
2. Builds lookup dicts (`doc_id -> rank`, `doc_id -> score`) for each method.
3. Passes only the ordered doc IDs to `RRFFuser.fuse` (RRF uses ranks only).
4. Returns the top `top_n` results as dicts containing the text, RRF rank and score, and each method's rank and score.

### `print_results` and `main`

`print_results` formats the table with f-string alignment and truncates long text to 70 characters. `main` builds the engine once and loops: read a query, read `k` (digits, else 60), search, print.

The `if __name__ == "__main__":` guard means importing the module (for example from a Flask app) does not start the console loop.

### Overall flow

```
query -> BM25 ranks all 50 docs  --+
                                   +--> each doc gets 1/(k+rank) from each list
query -> cosine ranks all 50 docs -+     -> contributions summed -> sorted by RRF score
```

## Notes

- If a query shares no words with any document, all BM25 scores are zero, so its ranks are just corpus order. RRF still works, but BM25 adds no real signal for that query.
- Both retrievers rank the full corpus, so every document appears in both lists.
