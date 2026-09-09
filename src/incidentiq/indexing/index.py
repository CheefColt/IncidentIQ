import math
from collections import Counter

import pandas as pd

from incidentiq.indexing.tokenizer import tokenize


class Index:
    """
    Builds the core inverted and positional indexes used by retrieval.

    Stores:
        - document lengths
        - average document length
        - positional index
        - inverted index
        - IDF scores
    """

    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.N = len(df)

        self.doc_lengths: dict[int, int] = {}

        self.positional_index: dict = {}
        self.inverted_index: dict = {}

        self.idf_scores: dict = {}

        self.avgdl = 0.0

        self._build()

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Build all index structures."""

        self._build_indexes()
        self._calculate_idf_scores()

    def _build_indexes(self) -> None:
        """
        Tokenize every document once and build:

        - document lengths
        - positional index
        - inverted index
        """

        total_length = 0

        for doc_id, row in self.df.iterrows():

            terms = tokenize(row["message"])

            # ----------------------------------------------------------
            # Document length
            # ----------------------------------------------------------

            doc_length = len(terms)

            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            # ----------------------------------------------------------
            # Term frequency
            # ----------------------------------------------------------

            term_counts = Counter(terms)

            for term, tf in term_counts.items():

                self.inverted_index \
                    .setdefault(term, {})[doc_id] = tf

            # ----------------------------------------------------------
            # Positional index
            # ----------------------------------------------------------

            for position, term in enumerate(terms):

                self.positional_index \
                    .setdefault(term, {}) \
                    .setdefault(doc_id, []) \
                    .append(position)

        # --------------------------------------------------------------
        # Average document length
        # --------------------------------------------------------------

        if self.N > 0:

            self.avgdl = total_length / self.N

        else:

            self.avgdl = 0.0

    def _calculate_idf_scores(self) -> None:
        """Calculate IDF for every indexed term."""

        for term in self.inverted_index:

            doc_freq = self.document_frequency(term)

            self.idf_scores[term] = self.idf(doc_freq)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def document_frequency(self, term: str) -> int:
        """Return the number of documents containing a term."""

        return len(
            self.positional_index.get(
                term.lower(),
                {}
            )
        )

    def idf(self, doc_freq: int) -> float:
        """
        Calculate BM25 IDF for a term.

        Uses the standard BM25 formulation:

            log(
                1 +
                (N - df + 0.5) /
                (df + 0.5)
            )
        """

        return math.log(
            1
            + (
                (self.N - doc_freq + 0.5)
                /
                (doc_freq + 0.5)
            )
        )