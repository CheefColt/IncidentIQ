## Defining a tool for the Agent to call for simple and direct retrieval search

import pandas as pd
from datetime import datetime

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

    print(results[["timestamp", "node", "severity", "message"]].head(10))

# print(search_logs(df, severity="INFO"))
# print(search_logs(df, severity="FATAL"))
# print(search_logs(df, severity="WARNING"))
# print(search_logs(df, severity="SEVERE"))
# print(search_logs(df, severity="ERROR"))
# print(search_logs(df, component="DISCOVERY"))
# print(search_logs(df, severity="ERROR", component="DISCOVERY"))
# print(search_logs(df))
# print(search_logs(df,severity="INFO",start_time=datetime.strptime("2005-06-28 09:53:39.479164","%Y-%m-%d %H:%M:%S.%f"),end_time=datetime.strptime("2005-08-02 21:15:36.811548","%Y-%m-%d %H:%M:%S.%f")))



print(f"{search_logs(df,query="instruction cache")}\n")
print(search_logs(df,query="core files"))
