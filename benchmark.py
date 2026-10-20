import time
import pandas as pd

from incidentiq.pipeline import IncidentIQ
from incidentiq.evaluation.agent_metrics import evaluate_agent

TEST_SUITE = [
    {
        "id": "INC-01",
        "query": "R02-M1-N0-C:J12-U11",
        "description": "Exact hardware node path (Tests custom tokenizer & BM25 recall)",
    },
    {
        "id": "INC-02",
        "query": "data TLB error interrupt",
        "description": "Kernel architecture error message (Tests semantic / symptom search)",
    },
    {
        "id": "INC-03",
        "query": "node R02 critical failure",
        "description": "Partial node identifier + symptom (Tests hybrid RRF fusion)",
    },
]


DATA_PATH = "data/processed/logs.parquet"



def run_benchmark():

    print("="*70)
    print("INCIDENT EVALUATION & BENCHMARK HARNESS")
    print("="*70)

    app = IncidentIQ(data_path=DATA_PATH)
    results = []

    for test_case in TEST_SUITE:
        query = test_case["query"]

        print(f"\n[Running Query] : {query} ({test_case['description']})")

        start = time.perf_counter()
        inv_result = app.investigate(query,top_k=10)
        duration = time.perf_counter() - start

        metrics = evaluate_agent(inv_result)

        results.append({
            "Query ID" : test_case["id"],
            "Query" : test_case["query"],
            "Latency (s)": round(duration,2),
            "Tool Calls" : len(inv_result.tool_calls),
            "Total Evidence": len(inv_result.evidence) + len(inv_result.tool_evidence),
            "Obs Citation" : f"{metrics['observation_citation_rate'] * 100:.0f}%",
            "Hyp Citation" : f"{metrics['hypothesis_citation_rate'] * 100:.0f}%",
            "Tool Yield" : metrics["tool_efficiency"] if metrics["tool_efficiency"] is not None else "N/A",
            "Consistent" : metrics["completion_consistent"],
            "Grounded" : metrics["grounded"]
        })

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))

    node_query = "R02-M1-N0-C:J12-U11"
    bm25_res = app.engine.search_bm25(node_query, top_k=10)
    sem_res = app.engine.search_semantic(node_query, top_k=10)
    hybrid_res = app.engine.search_hybrid(node_query, top_k=10)

        # Count how many returned logs actually belong to the target hardware node
    bm25_exact = sum(1 for r in bm25_res if r.get("node") == node_query)
    sem_exact = sum(1 for r in sem_res if r.get("node") == node_query)
    hybrid_exact = sum(1 for r in hybrid_res if r.get("node") == node_query)

    print("\n" + "=" * 70)
    print(f"RETRIEVAL PRECISION ON EXACT HARDWARE ID: '{node_query}'")
    print("=" * 70)
    print(f"BM25 Relevant Node Matches:     {bm25_exact} / 10")
    print(f"Semantic Relevant Node Matches: {sem_exact} / 10")
    print(f"Hybrid RRF Node Matches:        {hybrid_exact} / 10")



if __name__ == "__main__":
    run_benchmark()
