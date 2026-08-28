import math
from collections import Counter

import pandas as pd

from incidentiq.indexing.tokenizer import tokenize


class Index:

    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.N = len(df)

        self.doc_lengths = {}

        self.positional_index = {}
        self.inverted_index = {}

        self.idf_scores = {}

        self.avgdl = 0.0

        self._build()

    def _build(self):

        self._calculate_document_lengths()
        self._build_positional_index()
        self._build_inverted_index()
        self._calculate_idf_scores()

    def _calculate_document_lengths(self):

        for doc_id, row in self.df.iterrows():

            self.doc_lengths[doc_id] = len(
                tokenize(row["message"])
            )

        self.avgdl = (
            sum(self.doc_lengths.values())
            / self.N
        )

    def _build_positional_index(self):

        for doc_id, row in self.df.iterrows():

            terms = tokenize(row["message"])

            for position, term in enumerate(terms):

                self.positional_index \
                    .setdefault(term, {}) \
                    .setdefault(doc_id, []) \
                    .append(position)

    def _build_inverted_index(self):

        for doc_id, row in self.df.iterrows():

            terms = tokenize(row["message"])
            term_counts = Counter(terms)

            for term, tf in term_counts.items():

                self.inverted_index \
                    .setdefault(term, {})[doc_id] = tf

    def document_frequency(self, term: str) -> int:

        return len(
            self.positional_index.get(
                term.lower(),
                {}
            )
        )

    def idf(self, doc_freq: int) -> float:

        return math.log(
            1
            + (
                (self.N - doc_freq + 0.5)
                /
                (doc_freq + 0.5)
            )
        )

    def _calculate_idf_scores(self):

        for term in self.inverted_index:

            df = self.document_frequency(term)

            self.idf_scores[term] = self.idf(df)