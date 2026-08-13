## Defining a tool for the Agent to call for simple and direct retrieval search

import pandas as pd

df = pd.read_parquet(
        r"E:\incidentiq\data\processed\logs.parquet"
    )

def search_logs(df, severity=None):
    if severity is not None:
        return df[df["severity"]==severity]
    else:
        return df[df["severity"]]
            

# print(search_logs(df, severity="INFO"))
# print(search_logs(df, severity="FATAL"))
# print(search_logs(df, severity="WARNING"))
# print(search_logs(df, severity="SEVERE"))
print(search_logs(df, severity="ERROR"))

