
import math
import re
from collections import Counter

import pandas as pd


class Index:
    """
    Builds inverted and positional indexes for log messages and node IDs.

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

    def _document_terms(self, row: pd.Series) -> list[str]:
        """Combine searchable message and node identifier terms."""
        message_terms = tokenize(str(row["message"]))

        node = str(row.get("node", "")).strip().lower()
        node_terms = tokenize(node)

        # Add node components for partial identifier searches.
        # Example: R26-M0-N0-I:J18-U11 -> j18-u11, j18, u11, etc.
        node_parts = re.findall(r"[a-z0-9]+", node)

        terms = message_terms + node_terms + node_parts

        # Preserve order while avoiding redundant additions.
        return list(dict.fromkeys(terms))

    def _build(self) -> None:
        self._build_indexes()
        self._calculate_idf_scores()

    def _build_indexes(self) -> None:
        total_length = 0

        for doc_id, row in self.df.iterrows():
            terms = self._document_terms(row)

            doc_length = len(terms)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            term_counts = Counter(terms)

            for term, tf in term_counts.items():
                self.inverted_index.setdefault(term, {})[doc_id] = tf

            for position, term in enumerate(terms):
                self.positional_index.setdefault(term, {}).setdefault(
                    doc_id, []
                ).append(position)

        self.avgdl = total_length / self.N if self.N > 0 else 0.0

    def _calculate_idf_scores(self) -> None:
        for term in self.inverted_index:
            doc_freq = self.document_frequency(term)
            self.idf_scores[term] = self.idf(doc_freq)

    def document_frequency(self, term: str) -> int:
        return len(self.positional_index.get(term.lower(), {}))

    def idf(self, doc_freq: int) -> float:
        return math.log(
            1
            + (self.N - doc_freq + 0.5) / (doc_freq + 0.5)
        )
