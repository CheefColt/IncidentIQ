import pandas as pd

from incidentiq.indexing.index import Index
from incidentiq.retrieval.bm25 import BM25Retriever
from incidentiq.retrieval.query import (
    parse_query,
    get_scoring_terms
)
from incidentiq.retrieval.semantic import (
    SemanticRetriever
)
from incidentiq.ranking.rrf import (
    reciprocal_rank_fusion
)


class SearchEngine:

    def __init__(self, data_path: str):

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

    def search_bm25(
        self,
        query: str,
        top_k: int = 10
    ):

        parsed = parse_query(query)

        terms = get_scoring_terms(
            parsed
        )

        ranking = self.bm25.search(
            terms
        )

        return self._build_results(
            ranking[:top_k]
        )

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        semantic_k: int = 50
    ):

        parsed = parse_query(query)

        terms = get_scoring_terms(
            parsed
        )

        bm25_ranking = self.bm25.search(
            terms
        )

        semantic_ranking, _ = (
            self.semantic.search(
                query,
                semantic_k
            )
        )

        rrf_scores = reciprocal_rank_fusion(
            [
                bm25_ranking,
                semantic_ranking
            ]
        )

        ranking = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )

        return self._build_results(
            ranking[:top_k],
            rrf_scores
        )

    def _build_results(
        self,
        ranking,
        scores=None
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
                result["score"] = scores[
                    doc_id
                ]

            results.append(result)

        return results