import pandas as pd

from incidentiq.indexing.index import Index
from incidentiq.ranking.rrf import reciprocal_rank_fusion
from incidentiq.retrieval.bm25 import BM25Retriever
from incidentiq.retrieval.query import (
    get_scoring_terms,
    parse_query,
)
from incidentiq.retrieval.semantic import SemanticRetriever


DEFAULT_SEMANTIC_K = 50


class SearchEngine:
    """
    Main search interface for IncidentIQ.

    Coordinates:

        corpus
            ↓
        indexing
            ↓
        ┌───────────────┐
        │               │
       BM25         Semantic
        │               │
        └───────┬───────┘
                ↓
               RRF
                ↓
          final results
    """

    def __init__(
        self,
        data_path: str,
    ):
        """
        Load the corpus and initialize all
        retrieval components.
        """

        self.df = pd.read_parquet(
            data_path
        )

        self.index = Index(
            self.df
        )

        self.bm25 = BM25Retriever(
            self.index
        )

        self.semantic = SemanticRetriever(
            self.df
        )

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search using BM25 lexical retrieval.
        """

        parsed_query = parse_query(
            query
        )

        terms = get_scoring_terms(
            parsed_query
        )

        ranked_results = self.bm25.search(
            terms
        )

        top_results = ranked_results[
            :top_k
        ]

        doc_ids = [
            doc_id
            for doc_id, _ in top_results
        ]

        scores = {
            doc_id: score
            for doc_id, score in top_results
        }

        return self._build_results(
            doc_ids,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # SEMANTIC
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search using semantic similarity.
        """

        ranking, scores = self.semantic.search(
            query,
            top_k,
        )

        results = []

        for rank, doc_id in enumerate(
            ranking,
            start=1,
        ):

            row = self.df.loc[
                doc_id
            ]

            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "semantic_score": float(
                        scores[rank - 1]
                    ),
                    "log_id": row["log_id"],
                    "timestamp": row["timestamp"],
                    "node": row["node"],
                    "severity": row["severity"],
                    "message": row["message"],
                }
            )

        return results

    # ------------------------------------------------------------------
    # HYBRID / RRF
    # ------------------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        semantic_k: int = DEFAULT_SEMANTIC_K,
    ) -> list[dict]:
        """
        Search using Reciprocal Rank Fusion (RRF).

        BM25 contributes its complete ranking while
        semantic retrieval contributes its top
        `semantic_k` candidates.
        """

        parsed_query = parse_query(
            query
        )

        terms = get_scoring_terms(
            parsed_query
        )

        # --------------------------------------------------------------
        # BM25 ranking
        # --------------------------------------------------------------

        bm25_results = self.bm25.search(
            terms
        )

        bm25_ranking = [
            doc_id
            for doc_id, _ in bm25_results
        ]

        # --------------------------------------------------------------
        # Semantic ranking
        # --------------------------------------------------------------

        semantic_ranking, _ = (
            self.semantic.search(
                query,
                semantic_k,
            )
        )

        # --------------------------------------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------------------------------------

        rrf_scores = reciprocal_rank_fusion(
            [
                bm25_ranking,
                semantic_ranking,
            ]
        )

        final_ranking = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        return self._build_results(
            final_ranking[:top_k],
            scores=rrf_scores,
        )

    # ------------------------------------------------------------------
    # RESULT BUILDING
    # ------------------------------------------------------------------

    def _build_results(
        self,
        ranking: list[int],
        scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """
        Convert ranked document IDs into the
        public search-result format.
        """

        results = []

        for rank, doc_id in enumerate(
            ranking,
            start=1,
        ):

            row = self.df.loc[
                doc_id
            ]

            result = {
                "rank": rank,
                "doc_id": doc_id,
                "log_id": row["log_id"],
                "timestamp": row["timestamp"],
                "node": row["node"],
                "severity": row["severity"],
                "message": row["message"],
            }

            if scores is not None:

                result["score"] = float(
                    scores[doc_id]
                )

            results.append(
                result
            )

        return results