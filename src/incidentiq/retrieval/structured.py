import math
import re
from collections import Counter
from itertools import product
from sentence_transformers import SentenceTransformer, util

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"E:\incidentiq\data\processed\logs.parquet"

K1 = 1.2
B = 0.75

BM25_WEIGHT = 1.0
PHRASE_WEIGHT = 3.0
PROXIMITY_WEIGHT = 2.0


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert a log message into normalized searchable terms.

    Examples:

        "instruction cache parity error"
        -> ["instruction", "cache", "parity", "error"]

        "fpr29=0xffffffff"
        -> ["fpr29", "0xffffffff"]

        "172.16.96.116:33569"
        -> ["172.16.96.116:33569"]
    """

    return re.findall(
        r"[a-z0-9_./:-]+",
        text.lower()
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(DATA_PATH)


# ============================================================
# DOCUMENT LENGTH
# ============================================================

df["doc_length"] = df["message"].apply(
    lambda text: len(tokenize(text))
)

N = len(df)

AVGDL = df["doc_length"].mean()


# ============================================================
# POSITIONAL INDEX
#
# term -> {
#     doc_id -> [positions]
# }
#
# Example:
#
# positional_index["cache"]
#
# {
#     0: [1],
#     1: [1],
#     42: [3]
# }
# ============================================================

positional_index = {}


for doc_id, row in df.iterrows():

    terms = tokenize(row["message"])

    for position, term in enumerate(terms):

        positional_index \
            .setdefault(term, {}) \
            .setdefault(doc_id, []) \
            .append(position)


# ============================================================
# INVERTED INDEX
#
# term -> {
#     doc_id -> term_frequency
# }
#
# Example:
#
# inverted_index["error"]
#
# {
#     0: 1,
#     1: 1,
#     25: 3
# }
# ============================================================

inverted_index = {}


for doc_id, row in df.iterrows():

    terms = tokenize(row["message"])

    term_counts = Counter(terms)

    for term, tf in term_counts.items():

        inverted_index \
            .setdefault(term, {})[doc_id] = tf


# ============================================================
# DOCUMENT FREQUENCY
# ============================================================

def document_frequency(term: str) -> int:

    return len(
        positional_index.get(
            term.lower(),
            {}
        )
    )


# ============================================================
# IDF - Inverse Document Frequency
# ============================================================

def idf(doc_freq: int) -> float:

    return math.log(
        1
        + (
            (N - doc_freq + 0.5)
            /
            (doc_freq + 0.5)
        )
    )


# ============================================================
# PRECOMPUTE IDF
# ============================================================

idf_scores = {}

for term in inverted_index:

    df_score = document_frequency(term)

    idf_scores[term] = idf(df_score)


# ============================================================
# BM25 SCORING
# ============================================================

def bm25_score(
    doc_id: int,
    terms: list[str],
    doc_length: int
) -> float:

    score = 0.0

    for term in terms:

        term = term.lower()

        # Unknown query terms contribute nothing.
        if term not in idf_scores:
            continue

        tf = inverted_index[
            term
        ].get(doc_id, 0)

        if tf == 0:
            continue

        IDF = idf_scores[term]

        contribution = (
            IDF
            * (
                (tf * (K1 + 1))
                /
                (
                    tf
                    + K1
                    * (
                        1
                        - B
                        + B * (doc_length / AVGDL)
                    )
                )
            )
        )

        score += contribution

    return score


# ============================================================
# QUERY PARSER
#
# Example:
#
#     cache "parity error"
#
# becomes:
#
# {
#     "terms": ["cache"],
#     "phrases": [["parity", "error"]]
# }
# ============================================================

def parse_query(query: str) -> dict:

    terms = []
    phrases = []

    inside_quotes = False

    current_phrase = []
    current_text = []

    def flush_text():

        nonlocal current_text

        if not current_text:
            return

        text = "".join(current_text)

        tokens = tokenize(text)

        if inside_quotes:

            current_phrase.extend(tokens)

        else:

            terms.extend(tokens)

        current_text = []


    for char in query:

        # ----------------------------------------------------
        # Quote
        # ----------------------------------------------------

        if char == '"':

            if inside_quotes:

                # Finish the phrase.
                flush_text()

                if current_phrase:

                    phrases.append(
                        current_phrase
                    )

                current_phrase = []

            else:

                # Start a phrase.
                flush_text()

            inside_quotes = not inside_quotes


        # ----------------------------------------------------
        # Whitespace
        # ----------------------------------------------------

        elif char.isspace():

            flush_text()


        # ----------------------------------------------------
        # Normal character
        # ----------------------------------------------------

        else:

            current_text.append(char)


    # Flush anything remaining at the end.
    flush_text()


    return {
        "terms": terms,
        "phrases": phrases
    }


# ============================================================
# GET SCORING TERMS
#
# Normal terms + terms inside phrases
#
# Example:
#
# cache "parity error"
#
# ->
#
# ["cache", "parity", "error"]
# ============================================================

def get_scoring_terms(
    parsed_query: dict
) -> list[str]:

    scoring_terms = (
        parsed_query["terms"].copy()
    )

    for phrase in parsed_query["phrases"]:

        scoring_terms.extend(
            phrase
        )

    return scoring_terms


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def get_term_candidates(
    terms: list[str]
) -> set[int]:

    candidate_docs = set()

    for term in terms:

        candidate_docs.update(
            positional_index.get(
                term.lower(),
                {}
            )
        )

    return candidate_docs


def get_candidates(
    parsed_query: dict
) -> set[int]:

    scoring_terms = get_scoring_terms(
        parsed_query
    )

    return get_term_candidates(
        scoring_terms
    )


# ============================================================
# EXACT PHRASE SEARCH
#
# Example:
#
# ["parity", "error"]
#
# matches:
#
#     parity error
#
# but not:
#
#     parity was reported as an error
# ============================================================

def phrase_search(
    phrase_terms: list[str]
) -> set[int]:

    if not phrase_terms:
        return set()


    phrase_terms = [
        term.lower()
        for term in phrase_terms
    ]


    first_term = phrase_terms[0]

    candidate_docs = positional_index.get(
        first_term,
        {}
    )


    matches = set()


    for doc_id, positions in candidate_docs.items():

        for position in positions:

            match = True


            for offset, term in enumerate(
                phrase_terms[1:],
                start=1
            ):

                term_positions = (
                    positional_index
                    .get(term, {})
                    .get(doc_id, [])
                )


                if position + offset not in term_positions:

                    match = False

                    break


            if match:

                matches.add(doc_id)

                break


    return matches


# ============================================================
# CHECK WHETHER A DOCUMENT MATCHES A PHRASE
# ============================================================

def phrase_matches(
    doc_id: int,
    phrase: list[str]
) -> bool:

    return doc_id in phrase_search(
        phrase
    )


# ============================================================
# PHRASE SCORE
# ============================================================

def calculate_phrase_score(
    doc_id: int,
    phrases: list[list[str]]
) -> float:

    matched_phrases = 0


    for phrase in phrases:

        if phrase_matches(
            doc_id,
            phrase
        ):

            matched_phrases += 1


    return (
        PHRASE_WEIGHT
        * matched_phrases
    )


# ============================================================
# MINIMUM SPAN
#
# Finds the smallest window containing all query terms.
#
# Example:
#
# cache search parity error
#
# positions:
#
# cache  -> 1
# parity -> 3
# error  -> 4
#
# span = 4
# ============================================================

def minimum_span(
    doc_id: int,
    terms: list[str]
):

    positions = []


    for term in terms:

        term_positions = (
            positional_index
            .get(
                term.lower(),
                {}
            )
            .get(
                doc_id,
                []
            )
        )


        if not term_positions:

            return None


        positions.append(
            term_positions
        )


    min_span = float("inf")


    for combination in product(
        *positions
    ):

        start = min(
            combination
        )

        end = max(
            combination
        )

        span = (
            end
            - start
            + 1
        )


        if span < min_span:

            min_span = span


    return min_span


# ============================================================
# PROXIMITY SCORE
#
# Perfect adjacency:
#
#     cache parity error
#     -> 1.0
#
# One extra token:
#
#     cache search parity error
#     -> 0.5
#
# Two extra tokens:
#
#     -> 0.333...
# ============================================================

def proximity_score(
    doc_id: int,
    terms: list[str]
):

    span = minimum_span(
        doc_id,
        terms
    )


    # One or more query terms are missing.
    if span is None:

        return None


    # Number of distinct query terms represents
    # the theoretical minimum span.
    minimum_possible_span = len(
        set(terms)
    )


    extra_distance = (
        span
        - minimum_possible_span
    )


    return 1 / (
        1 + extra_distance
    )


# ============================================================
# FINAL SEARCH / RANKING
# ============================================================

def search(
    query: str,
    top_k: int = 10
) -> list[dict]:

    # --------------------------------------------------------
    # Parse query
    # --------------------------------------------------------

    parsed_query = parse_query(
        query
    )


    # --------------------------------------------------------
    # Extract terms used for scoring
    # --------------------------------------------------------

    scoring_terms = get_scoring_terms(
        parsed_query
    )


    # Empty query.
    if not scoring_terms:

        return []


    # --------------------------------------------------------
    # Retrieve candidate documents
    # --------------------------------------------------------

    candidate_docs = get_candidates(
        parsed_query
    )


    results = []


    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    for doc_id in candidate_docs:

        row = df.loc[doc_id]


        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25 = bm25_score(
            doc_id,
            scoring_terms,
            row["doc_length"]
        )


        # ----------------------------------------------------
        # Phrase score
        # ----------------------------------------------------

        phrase_score = calculate_phrase_score(
            doc_id,
            parsed_query["phrases"]
        )


        # ----------------------------------------------------
        # Proximity
        # ----------------------------------------------------

        proximity = proximity_score(
            doc_id,
            scoring_terms
        )


        # A document missing one or more query terms
        # receives no proximity contribution.
        if proximity is None:

            proximity = 0.0


        # ----------------------------------------------------
        # Final ranking score
        # ----------------------------------------------------

        final_score = (
            BM25_WEIGHT * bm25
            + phrase_score
            + PROXIMITY_WEIGHT * proximity
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "score": final_score,

            "bm25": bm25,

            "phrase_score": phrase_score,

            "proximity": proximity,

            "log_id": row["log_id"],

            "timestamp": row["timestamp"],

            "node": row["node"],

            "severity": row["severity"],

            "message": row["message"]
        })


    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    return results[:top_k]


# ============================================================
# TESTING
# ============================================================

# if __name__ == "__main__":

#     test_queries = [

#         "cache error",

#         "parity error",

#         '"parity error"',

#         'cache "parity error"',

#         "core files",
#     ]


#     for query in test_queries:

#         print()
#         print("=" * 80)
#         print(f"QUERY: {query}")
#         print("=" * 80)


#         results = search(
#             query,
#             top_k=3
#         )


#         for result in results:

#             print(
#                 f"{result['score']:.2f} | "
#                 f"BM25={result['bm25']:.2f} | "
#                 f"Phrase={result['phrase_score']:.2f} | "
#                 f"Proximity={result['proximity']:.2f}"
#             )

#             print(
#                 f"    {result['message']}"
#             )

# ==========================================
# Evaluating Retreival Quality
# ==========================================

# query = "cache parity error"
# results = search(query, top_k=10)
# results = search("cache problem", top_k=10)

# for rank, result in enumerate(results, start=1):
#     print(
#         f"{rank:2}. "
#         f"{result['score']:.2f} | "
#         f"log_id={result['log_id']} | "
#         f"{result['timestamp']} | "
#         f"{result['message']}"
#     )



# ============================================
# Vector Embeddings
# ============================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

log_embeddings = model.encode(
    df["message"].tolist(),
    show_progress_bar=True
)
def min_max_normalize(scores):

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [
        (score-min_score) / (max_score - min_score)
        for score in scores
    ]

def get_semantic_candidates(query:str, top_k:int=50):
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        log_embeddings
    )[0]

    top_scores, top_indices = scores.topk(top_k)

    candidate_docs = {
        int(doc_id)
        for doc_id in top_indices
    }

    return candidate_docs, scores

def hybrid_search(query:str, top_k:int=10, semantic_k:int=50):

    parsed = parse_query(query)

    scoring_terms = get_scoring_terms(parsed)

    # BM25 candidates
    bm25_candidates = get_candidates(parsed)

    # Semantic Candidates
    semantic_candidates, semantic_scores = (get_semantic_candidates(
        query,
        semantic_k
    ))


    # Combines cadidates

    candidate_docs = bm25_candidates | semantic_candidates

    print(
        f"BM25 candidates: "
        f"{len(bm25_candidates)}"
    )

    print(
        f"Semantic candidates: "
        f"{len(semantic_candidates)}"
    )

    print(
        f"Combined candidates: "
        f"{len(candidate_docs)}"
    )

    # # Semantic representation of query
    # query_embedding = model.encode(
    #     query,
    #     convert_to_tensor=True
    # )

    # semantic_scores = util.cos_sim(
    #     query_embedding,
    #     log_embeddings
    # )[0]

    results = []

    for doc_id in candidate_docs:

        row = df.loc[doc_id]

        bm25 = bm25_score(
            doc_id,
            scoring_terms,
            row["doc_length"]
        )

        semantic_score = semantic_scores[doc_id].item()

        results.append({
            "doc_id": doc_id,
            "bm25": bm25,
            "semantic": semantic_score,
            "log_id": row["log_id"],
            "timestamp": row["timestamp"],
            "node": row["node"],
            "severity": row["severity"],
            "message": row["message"]
        })

    # Normalize

    bm25_values = [
        result["bm25"] for result in results
    ]

    semantic_values = [
        result["semantic"] for result in results
    ]

    bm25_norm = min_max_normalize(bm25_values)

    semantic_norm = min_max_normalize(
        semantic_values
    )

    ALPHA = 0.6

    for result, bm25, semantic in zip(results, bm25_norm, semantic_norm):

        result["bm25_norm"] = bm25

        result["semantic_norm"] = semantic

        result["score"] = (
            ALPHA * bm25 + (1 - ALPHA) * semantic
        )

    results.sort(key= lambda x: x["score"], reverse=True)

    return results

# results = hybrid_search(
#     "cache problem",
#     top_k=10
# )

# bm25_scores = [
#     result["bm25"]
#     for result in results
# ]

# semantic_scores_list = [
#     result["semantic"]
#     for result in results
# ]

# normalized_bm25 = min_max_normalize(bm25_scores)
# normalized_semantic = min_max_normalize(semantic_scores_list)

# for result, bm25, semantic in zip(
#     results,
#     normalized_bm25,
#     normalized_semantic
# ):
#     result["bm25_norm"] = bm25
#     result["semantic_norm"] = semantic

# ALPHA = 0.6

# for result in results:
#     result["score"] = ( ALPHA * result["bm25_norm"] + (1 - ALPHA) * result["semantic_norm"])

# results.sort(key = lambda x : x["score"], reverse=True)

# for result in results[:10]:
#     print(
#         f"{result['bm25_norm']:.3f} | "
#         f"{result['semantic_norm']:.3f} | "
#         f"{result['score']:.3f} | "
#         f"{result['message']}"
#     )



# results = hybrid_search(
#     "cache problem",
#     top_k=10
# )

# for result in results:

#     print(
#         f"{result['score']:.3f} | "
#         f"BM25={result['bm25_norm']:.3f} | "
#         f"SEM={result['semantic_norm']:.3f} | "
#         f"{result['message']}"
#     )

#################################
# Search v/s Hybrid Search
#################################


queries = [
    "cache error", "parity error", "core files", "machine malfucntion", "network connection failure"
]

for query in queries:

    search_results = search(
        query,
        top_k=10
    )

    hybrid_search_results = hybrid_search(
        query,
        top_k=10
    )

    print("#"*10+f" {query} "+"#"*10)

    print("\nNormal Search\n")
    
    for result in search_results:
        print(
        f"{result['score']:.3f} |"
        f"{result['message']}"
      )

    print("\nHybrid Search\n")

    for result in hybrid_search_results:
        print(
            f"{result['score']:.3f} | "
            f"BM25={result['bm25_norm']:.3f} | "
            f"SEM={result['semantic_norm']:.3f} | "
            f"{result['message']}"
        )

    print("\n")