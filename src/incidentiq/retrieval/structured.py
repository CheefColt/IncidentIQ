from incidentiq.search import SearchEngine


DEFAULT_DATA_PATH = "data/processed/logs.parquet"


def print_bm25_results(
    results: list[dict],
) -> None:
    """Print BM25 search results."""

    print("\nBM25 Search\n")

    for result in results:

        print(
            f"{result['rank']:2}. "
            f"{result.get('score', 0):.5f} | "
            f"{result['message']}"
        )


def print_semantic_results(
    results: list[dict],
) -> None:
    """Print semantic search results."""

    print("\nSemantic Search\n")

    for result in results:

        print(
            f"{result['rank']:2}. "
            f"{result.get('semantic_score', 0):.5f} | "
            f"{result['message']}"
        )


def print_hybrid_results(
    results: list[dict],
) -> None:
    """Print hybrid RRF search results."""

    print("\nHybrid Search (RRF)\n")

    for result in results:

        print(
            f"{result['rank']:2}. "
            f"{result.get('score', 0):.5f} | "
            f"{result['message']}"
        )


def run_search_demo(
    query: str,
    data_path: str = DEFAULT_DATA_PATH,
    top_k: int = 10,
) -> None:
    """
    Run BM25, semantic, and hybrid retrieval
    for a single query and display the results.
    """

    engine = SearchEngine(
        data_path
    )

    bm25_results = engine.search_bm25(
        query,
        top_k=top_k,
    )

    semantic_results = engine.search_semantic(
        query,
        top_k=top_k,
    )

    hybrid_results = engine.search_hybrid(
        query,
        top_k=top_k,
    )

    print_bm25_results(
        bm25_results
    )

    print_semantic_results(
        semantic_results
    )

    print_hybrid_results(
        hybrid_results
    )


def main() -> None:
    """Run the default search demonstration."""

    query = "hardware stopped working"

    run_search_demo(
        query=query,
        top_k=10,
    )


if __name__ == "__main__":
    main()