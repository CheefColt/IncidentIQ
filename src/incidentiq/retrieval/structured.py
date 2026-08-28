from incidentiq.search import SearchEngine


DATA_PATH = "data/processed/logs.parquet"


engine = SearchEngine(
    DATA_PATH
)


query = "hardware stopped working"


results = engine.search_hybrid(
    query,
    top_k=10
)


for result in results:

    print(
        f"{result['rank']:2}. "
        f"{result['score']:.5f} | "
        f"{result['message']}"
    )