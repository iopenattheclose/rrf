from flask import Flask, jsonify, render_template, request

from reciprocal_rank_fusion import DOCUMENTS, HybridSearch

app = Flask(__name__)

# Build the engine once at startup (loads the embedding model and embeds the docs).
engine = HybridSearch(DOCUMENTS)

K_MIN, K_MAX, K_DEFAULT = 1, 200, 60
TOP_N = 10


@app.route("/")
def index():
    return render_template("index.html", k_min=K_MIN, k_max=K_MAX, k_default=K_DEFAULT)


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()

    try:
        k = int(request.args.get("k", K_DEFAULT))
    except ValueError:
        k = K_DEFAULT
    k = max(K_MIN, min(k, K_MAX))

    if not query:
        return jsonify({"query": "", "k": k, "bm25": [], "cosine": [], "rrf": []})

    results = engine.search(query, k=k, top_n=TOP_N)
    return jsonify({"query": query, "k": k, **results})


if __name__ == "__main__":
    # use_reloader=False so the embedding model is not loaded twice.
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)