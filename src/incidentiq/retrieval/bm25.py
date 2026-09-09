class BM25Retriever:
    """
    BM25 lexical retriever.

    Uses the inverted index and document statistics
    provided by the Index class.
    """

    def __init__(
        self,
        index,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.index = index

        self.k1 = k1
        self.b = b

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------

    def score(
        self,
        doc_id: int,
        terms: list[str],
    ) -> float:
        """
        Calculate the BM25 score for one document.

        Parameters
        ----------
        doc_id:
            Document being scored.

        terms:
            Tokenized query terms.

        Returns
        -------
        float
            BM25 relevance score.
        """

        score = 0.0

        doc_length = self.index.doc_lengths[doc_id]

        for term in terms:

            term = term.lower()

            term_data = self.index.inverted_index.get(
                term,
                {}
            )

            tf = term_data.get(
                doc_id,
                0
            )

            # The document does not contain this term.
            if tf == 0:
                continue

            idf = self.index.idf_scores.get(
                term,
                0.0
            )

            numerator = (
                tf * (self.k1 + 1)
            )

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (
                        doc_length
                        / self.index.avgdl
                    )
                )
            )

            score += (
                idf
                * (
                    numerator
                    / denominator
                )
            )

        return score

    # ------------------------------------------------------------------
    # CANDIDATES
    # ------------------------------------------------------------------

    def candidates(
        self,
        terms: list[str],
    ) -> set[int]:
        """
        Return documents containing at least one
        of the query terms.
        """

        candidate_docs: set[int] = set()

        for term in terms:

            candidate_docs.update(
                self.index.inverted_index.get(
                    term.lower(),
                    {}
                )
            )

        return candidate_docs

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        terms: list[str],
    ) -> list[tuple[int, float]]:
        """
        Retrieve and rank documents using BM25.

        Returns
        -------
        list[tuple[int, float]]
            Ranked pairs of:

                (doc_id, BM25 score)
        """

        candidates = self.candidates(
            terms
        )

        doc_scores = {
            doc_id: self.score(
                doc_id,
                terms
            )
            for doc_id in candidates
        }

        ranked_docs = sorted(
            candidates,
            key=lambda doc_id: doc_scores[doc_id],
            reverse=True,
        )

        return [
            (
                doc_id,
                doc_scores[doc_id]
            )
            for doc_id in ranked_docs
        ]