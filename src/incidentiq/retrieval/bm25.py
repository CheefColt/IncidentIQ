class BM25Retriever:

    def __init__(
        self,
        index,
        k1: float = 1.2,
        b: float = 0.75
    ):
        self.index = index

        self.k1 = k1
        self.b = b

    def score(
        self,
        doc_id: int,
        terms: list[str]
    ) -> float:

        score = 0.0

        doc_length = self.index.doc_lengths[doc_id]

        for term in terms:

            term = term.lower()

            # How many times does this term
            # occur in this document?
            tf = (
                self.index.inverted_index
                .get(term, {})
                .get(doc_id, 0)
            )

            # Term doesn't occur in this document.
            if tf == 0:
                continue

            # How rare is this term across the corpus?
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

            score += idf * (
                numerator / denominator
            )

        return score

    def candidates(
        self,
        terms: list[str]
    ) -> set[int]:

        candidate_docs = set()

        for term in terms:

            candidate_docs.update(
                self.index.inverted_index.get(
                    term.lower(),
                    {}
                )
            )

        return candidate_docs

    def search(
        self,
        terms: list[str]
    ) -> list[tuple[int, float]]:

        candidates = self.candidates(
            terms
        )

        doc_scores = {
            doc_id: self.score(doc_id, terms)
            for doc_id in candidates
        }

        ranked_docs = sorted(
            candidates,
            key=lambda doc_id: doc_scores[doc_id],
            reverse=True
        )

        return [
            (
                doc_id,
                doc_scores[doc_id]
            )
            for doc_id in ranked_docs
        ]