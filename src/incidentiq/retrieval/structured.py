## Defining a tool for the Agent to call for simple and direct retrieval search

import pandas as pd
from datetime import datetime
import math

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

def document_frequency(term):
    return df["message"].str.contains(term,case=False,na=False).sum()
    

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

def bm25score(document:str,query:str):
    score = 0
    k1 = 1.2
    b = 0.75

    for term in query.split():
        DF = document_frequency(term)
        IDF = idf(DF)
        TF = document.lower().split().count(term.lower())
        dl = len(document.split())
        bm25_contribution_of_term = (
            IDF * (
                (TF*(k1+1)) / (TF+ k1 * (1 - b + b * (dl/avgdl)))
            )
        )

        print(
            term,
            "DF =", DF,
            "IDF =", IDF,
            "TF =", TF,
            "DL =", dl,
            "contribution =", bm25_contribution_of_term
        )

        score += bm25_contribution_of_term

    return score


document = "instruction cache parity error corrected"
query = "cache error"

print(bm25score(document, query))