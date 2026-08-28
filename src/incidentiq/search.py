import pandas as pd

from incidentiq.indexing.index import Index

from incidentiq.retrieval.bm25 import (
    BM25Retriever
)

from incidentiq.retrieval.query import (
    parse_query,
    get_scoring_terms,
)

from incidentiq.retrieval.semantic import (
    SemanticRetriever
)

from incidentiq.ranking.rrf import (
    reciprocal_rank_fusion
)


class SearchEngine:

    def __init__(self, data_path: str):

        # -----------------------------------------
        # Load corpus
        # -----------------------------------------

        self.df = pd.read_parquet(
            data_path
        )

        # -----------------------------------------
        # Build indexes
        # -----------------------------------------

        self.index = Index(
            self.df
        )

        # -----------------------------------------
        # Create retrievers
        # -----------------------------------------

        self.bm25 = BM25Retriever(
            self.index
        )

        self.semantic = SemanticRetriever(
            self.df
        )

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    def search_bm25(
        self,
        query: str,
        top_k: int = 10
    ):

        parsed_query = parse_query(
            query
        )

        terms = get_scoring_terms(
            parsed_query
        )

        bm25_results = self.bm25.search(
            terms
        )

        doc_ids = [
            doc_id
            for doc_id, _ in bm25_results[:top_k]
        ]

        scores = {
            doc_id: score
            for doc_id, score in bm25_results[:top_k]
        }

        return self._build_results(
            doc_ids,
            scores=scores
        )

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    def search_semantic(
        self,
        query: str,
        top_k: int = 10
    ):

        ranking, scores = (
            self.semantic.search(
                query,
                top_k
            )
        )

        results = []

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            row = self.df.loc[doc_id]

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "semantic_score": float(
                    scores[rank - 1]
                ),
                "log_id": row["log_id"],
                "timestamp": row["timestamp"],
                "node": row["node"],
                "severity": row["severity"],
                "message": row["message"]
            })

        return results

    # =====================================================
    # HYBRID / RRF SEARCH
    # =====================================================

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        semantic_k: int = 50
    ):

        parsed_query = parse_query(
            query
        )

        terms = get_scoring_terms(
            parsed_query
        )

        # -----------------------------------------
        # Get independent rankings
        # -----------------------------------------

        bm25_results = self.bm25.search(
            terms
        )
        
        bm25_ranking = [
            doc_id
            for doc_id, _ in bm25_results
        ]

        semantic_ranking, _ = (
            self.semantic.search(
                query,
                semantic_k
            )
        )

        # -----------------------------------------
        # Combine rankings using RRF
        # -----------------------------------------

        rrf_scores = reciprocal_rank_fusion(
            [
                bm25_ranking,
                semantic_ranking
            ]
        )

        final_ranking = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )

        return self._build_results(
            final_ranking[:top_k],
            scores=rrf_scores
        )

    # =====================================================
    # RESULT BUILDING
    # =====================================================

    def _build_results(
        self,
        ranking: list[int],
        scores: dict[int, float] | None = None
    ):

        results = []

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            row = self.df.loc[doc_id]

            result = {
                "rank": rank,
                "doc_id": doc_id,
                "log_id": row["log_id"],
                "timestamp": row["timestamp"],
                "node": row["node"],
                "severity": row["severity"],
                "message": row["message"]
            }

            if scores is not None:

                result["score"] = float(
                    scores[doc_id]
                )

            results.append(result)

        return results