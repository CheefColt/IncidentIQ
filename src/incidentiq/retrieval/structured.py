## Defining a tool for the Agent to call for simple and direct retrieval search

import pandas as pd
from datetime import datetime
import math
from collections import Counter, defaultdict
import re

df = pd.read_parquet(
        r"E:\incidentiq\data\processed\logs.parquet"
    )

def search_logs(df, severity=None, component=None, start_time=None, end_time=None, query=None):
    # if severity is not None and component is None:
    #     return df[df["severity"]==severity]
    # elif severity is None and component is not None:
    #     return df[df["component"]==component]
    # elif severity is not None and component is not None:
    #     return df[(df["severity"]==severity) & (df["component"]==component)]
    # else:
    #     return df
    results = df

    if severity:
        results = results[results["severity"] == severity]

    if component:
        results = results[results["component"] == component]

    if start_time:
        results = results[results["timestamp"] >= start_time]

    if end_time:
        results = results[results["timestamp"] <= end_time]

    if query is not None:
        results = results[
            results["message"].str.contains(
                query,
                case=False,
                na=False
            )
        ]

    return results[["timestamp","message"]].head(10)

# print(search_logs(df, severity="INFO"))
# print(search_logs(df, severity="FATAL"))
# print(search_logs(df, severity="WARNING"))
# print(search_logs(df, severity="SEVERE"))
# print(search_logs(df, severity="ERROR"))
# print(search_logs(df, component="DISCOVERY"))
# print(search_logs(df, severity="ERROR", component="DISCOVERY"))
# print(search_logs(df))
# print(search_logs(df,severity="INFO",start_time=datetime.strptime("2005-06-28 09:53:39.479164","%Y-%m-%d %H:%M:%S.%f"),end_time=datetime.strptime("2005-08-02 21:15:36.811548","%Y-%m-%d %H:%M:%S.%f")))



# print(f"{search_logs(df,query="instruction cache")}\n")
# print(search_logs(df,query="core files"))
# print(search_logs(df,query="parity error"))

# Number of documents in the corpus
N = len(df)

# def document_frequency(term):
#     return df["message"].str.contains(term,case=False,na=False).sum()
    

def idf(doc_freq):
    return math.log(1+(N-doc_freq+0.5)/(doc_freq+0.5))

def term_frequency(term):
    return df.loc[
        df["message"].str.contains(term,case=False,na=False),
        ["message"]
    ].assign(
        term_freq=lambda x: x["message"].str.lower().str.count(term.lower())
    )


# for term in ["error", "cache", "parity", "KERNDTLB", "kernel"]:
#     doc_freq = document_frequency(term)
#     inverse_doc_freq = idf(doc_freq)
#     term_freq = term_frequency(term)
#     print(f"\nTerm: {term}, DF: \n{doc_freq}, \nIDF: {inverse_doc_freq}, \n {term_freq["term_freq"].value_counts().sort_index()} \n")

df["doc_length"] = df["message"].str.split().str.len()

# print(df["doc_length"].describe())

avgdl = df["doc_length"].mean()

# print("Average document length: ", avgdl)

# print(df[["message","doc_length"]].head(-10))

# print(df)

# Initial BM25 Score Function

# def BM25score(document:str,query:str):
#     score = 0
#     k1 = 1.2
#     b = 0.75

#     for term in query.split():
#         DF = document_frequency(term)
#         IDF = idf(DF)
#         TF = document.lower().split().count(term.lower())
#         dl = len(document.split())
#         bm25_contribution_of_term = (
#             IDF * (
#                 (TF*(k1+1)) / (TF+ k1 * (1 - b + b * (dl/avgdl)))
#             )
#         )

#         # print(
#         #     term,
#         #     "DF =", DF,
#         #     "IDF =", IDF,
#         #     "TF =", TF,
#         #     "DL =", dl,
#         #     "contribution =", bm25_contribution_of_term
#         # )

#         score += bm25_contribution_of_term

#     return score


# document = "instruction cache parity error corrected"
# query = "cache error"

# print(BM25score(document, query))

# def bm25_search(query:str,top_k:int=10):
#     results = []

#     for _, row in df.iterrows():

#         document = row["message"]
#         score = BM25score(document,query)

#         results.append(
#             {"score": score,
#              "log_id": row["log_id"],
#              "timestamp": row["timestamp"],
#              "node": row["node"],
#              "severity": row["severity"],
#              "message": row["message"]
#             }
#         )

#     results.sort(key=lambda x: x["score"], reverse=True)
#     return results[:top_k]

# print(BM25_search("cache error"))

## Learn Inverted Index

inverted_index = {}

# for doc_id, row in df.iterrows():

#     document = row["message"]

#     terms = document.lower().split()

#     for term in terms:
#         inverted_index.setdefault(term,set()).add(doc_id)

## Updating Inverted Index to store term : {doc_id : TF of term}
for doc_id, row in df.iterrows():

    document:str = row["message"]

    terms = document.lower().split()
    term_counts = Counter(terms)

    for term, tf in term_counts.items():
        inverted_index.setdefault(term,{})[doc_id] = tf


# print(inverted_index["cache"])
# print(inverted_index["error"])

query = "cache error"

# candidate_docs = set()

# for term in query.split():
#     candidate_docs.update(inverted_index[term])

# print(candidate_docs)
# print(len(candidate_docs))

# Updating Document Frequency function, after adding positional index
def document_frequency(term):
    return len(positional_index.get(term.lower(), {}))

# Created an index for IDF
idf_scores = {}

for term in inverted_index:
    doc_freq = document_frequency(term)
    idf_scores[term] = idf(doc_freq)


# print(idf_scores["cache"])
# print(idf_scores["error"])
# print(idf_scores["kernel"])


def updated_BM25score(doc_id:int,query:str,dl:int):
    # Update with added postional index
    score = 0
    k1 = 1.2
    b = 0.75

    for term in query.lower().split():
        IDF = idf_scores[term]
        positions = positional_index[term].get(doc_id, [])
        TF = len(positions)
        bm25_contribution_of_term = (
            IDF * (
                (TF*(k1+1)) / (TF+ k1 * (1 - b + b * (dl/avgdl)))
            )
        )

        score += bm25_contribution_of_term

    return score

def updated_BM25_search(query:str,top_k:int=10):

    # Update candidate retival, after addition of positional index
    candidate_docs = set()
    for term in query.lower().split():
        candidate_docs.update(positional_index.get(term, {}))

    print("Candidates:", len(candidate_docs))

    results = []

    for doc_id in candidate_docs:

        row = df.loc[doc_id]
        score = updated_BM25score(doc_id,query,row["doc_length"])
        
        results.append(
            {"score": score,
             "log_id": row["log_id"],
             "timestamp": row["timestamp"],
             "node": row["node"],
             "severity": row["severity"],
             "message": row["message"]
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# print(updated_BM25_search(query="parity error"))

# print("Rows:", len(df))

# print("DF(error):", document_frequency("error"))
# print("Index(error):", len(inverted_index.get("error", set())))

# print("DF(cache):", document_frequency("cache"))
# print("Index(cache):", len(inverted_index.get("cache", set())))
# print(updated_BM25_search(query="cache error"))

# for row in df[["log_id","message"]].head(20).itertuples():
#     print(row.log_id, row.message)

def tokenize(text:str):
    text = text.lower()

    return re.findall(
        r"[a-z0-9_./:-]+",
        text
    )

# tests = [
#     "instruction cache parity error corrected",
#     "double-hummer alignment exceptions",
#     "CE sym 2, at 0x0b5eee0, mask 0x05",
#     "CioStream socket to 172.16.96.116:33569",
#     "generating core.862",
#     "R02-M1-N0-C:J12-U11",
#     "fpr29=0xffffffff",
#     "/g/g24/germann2/SPaSM_mini/MEAM/r13"
# ]

# for text in tests:
#     print(text)
#     print(tokenize(text))
#     print()

# documents = {
#     0: "instruction cache parity error",
#     1: "cache parity instruction error"
# }

# positional_index = {}

# for doc_id, document in documents.items():

#     terms = document.lower().split()

#     for position, term in enumerate(terms):
#         positional_index.setdefault(term,{}).setdefault(doc_id,[]).append(position)


# def phrase_search(query:str, positional_index:dict):
#     query_terms = query.lower().split()

#     if not query_terms:
#         return []

#     first_term = query_terms[0]

#     candidate_docs = positional_index.get(first_term,[])

#     matches = []

#     for doc_id, positions in candidate_docs.items():
#         for position in positions:
#             match = True

#             for offset, term in enumerate(query_terms[1:],start=1):
#                 term_positions = positional_index.get(term,{}).get(doc_id,[])

#                 if position + offset not in term_positions:
#                     match = False
#                     break

#             if match:
#                 matches.append(doc_id)
#                 break

#     return matches

# print(positional_index)
# print(phrase_search("instruction cache", positional_index))
# print(phrase_search("cache parity", positional_index))
# print(phrase_search("parity instruction", positional_index))

positional_index = {}

for doc_id, row in df.iterrows():
    terms = tokenize(row["message"])

    for position, term in enumerate(terms):

        positional_index \
            .setdefault(term, {}) \
            .setdefault(doc_id, []) \
            .append(position)