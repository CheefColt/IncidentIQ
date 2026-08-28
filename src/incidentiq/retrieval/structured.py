from incidentiq.search import SearchEngine


DATA_PATH = "data/processed/logs.parquet"


engine = SearchEngine(DATA_PATH)


query = "hardware stopped working"


print("\nBM25 Search\n")

bm25_results = engine.search_bm25(
    query,
    top_k=10
)

for result in bm25_results:

    print(
        f"{result['rank']:2}. "
        f"{result.get('score', 0):.2f} | "
        f"{result['message']}"
    )


print("\nSemantic Search\n")

semantic_results = engine.search_semantic(
    query,
    top_k=10
)

for result in semantic_results:

    print(
        f"{result['rank']:2}. "
        f"{result['semantic_score']:.2f} | "
        f"{result['message']}"
    )


print("\nHybrid Search (RRF)\n")

hybrid_results = engine.search_hybrid(
    query,
    top_k=10
)

for result in hybrid_results:

    print(
        f"{result['rank']:2}. "
        f"{result['score']:.5f} | "
        f"{result['message']}"
    )