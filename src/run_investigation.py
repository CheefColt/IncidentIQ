from incidentiq.pipeline import IncidentIQ

def main():

    print("Initializing IncidentIQ pipeline.....")

    pipeline = IncidentIQ(data_path="../data/processed/logs.parquet")

    query = "critical failure or network erroe in node R02"
    print(f"Running investigation for query: {query!r}")

    result = pipeline.investigate(query=query, top_k=5)

    print("\n--- Investigation Finished ---")
    print(f"Summary: {result.analysis.summary}")
    print(f"Completed: {result.analysis.investigation_complete}")
    print(f"Tool calls executed: {result.tool_calls}")
    print(f"Total evidence gathered: {len(result.evidence)}")

if __name__=="__main__":
    main()