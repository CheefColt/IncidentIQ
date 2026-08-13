## Defining a tool for the Agent to call for simple and direct retrieval search

import pandas as pd

df = pd.read_parquet(
        r"E:\incidentiq\data\processed\logs.parquet"
    )

def search_logs(df, severity=None, component=None):
    if severity is not None and component is None:
        return df[df["severity"]==severity]
    elif severity is None and component is not None:
        return df[df["component"]==component]
    elif severity is not None and component is not None:
        return df[(df["severity"]==severity) & (df["component"]==component)]
    else:
        return df

            

# print(search_logs(df, severity="INFO"))
# print(search_logs(df, severity="FATAL"))
# print(search_logs(df, severity="WARNING"))
# print(search_logs(df, severity="SEVERE"))
print(search_logs(df, severity="ERROR"))
print(search_logs(df, component="DISCOVERY"))
print(search_logs(df, severity="ERROR", component="DISCOVERY"))
print(search_logs(df))



