class LogSearchTool:

    def __init__(self, engine):
        self.engine = engine

    def search(self, query: str, top_k: int = 10):
        return self.engine.search_hybrid(
            query=query,
            top_k=top_k,
        )