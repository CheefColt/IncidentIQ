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

            tf = self.index.inverted_index \
                .get(term, {}) \
                .get(doc_id, 0)

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

            score += idf * (
                numerator / denominator
            )

        return score

    def candidates(
        self,
        terms: list[str]
    ) -> set[int]:

        candidates = set()

        for term in terms:

            candidates.update(
                self.index.inverted_index.get(
                    term.lower(),
                    {}
                )
            )

        return candidates

    def search(
        self,
        terms: list[str]
    ) -> list[int]:

        candidates = self.candidates(terms)

        ranked = sorted(
            candidates,
            key=lambda doc_id: self.score(
                doc_id,
                terms
            ),
            reverse=True
        )

        return ranked