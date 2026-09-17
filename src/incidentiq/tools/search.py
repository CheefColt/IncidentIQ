class LogSearchTool:

    def __init__(self, engine):
        self.engine = engine

    def search(self, query: str, top_k: int = 10):
        results =  self.engine.search_hybrid(
            query=query,
            top_k=top_k,
        )

        evidence = []

        for result in results:

            doc_id = result["doc_id"]
            row = self.engine.df.loc[doc_id]

            evidence.append({
                "doc_id": doc_id,
                "timestamp": row["timestamp"],
                "node": row["node"],
                "severity": row["severity"],
                "message": row["message"]
            })

        return evidence