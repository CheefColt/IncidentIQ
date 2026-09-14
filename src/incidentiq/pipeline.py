from incidentiq.context.builder import ContextBuilder
from incidentiq.context.patterns import extract_patterns
from incidentiq.reasoning.analyzer import IncidentAnalyzer
from incidentiq.search import SearchEngine

class IncidentIQ:

    def __init__(
        self,
        data_path: str,
        model: str = "gemini-3.6-flash"
    ):

        self.engine = SearchEngine(data_path)
        self.context_builder = ContextBuilder(self.engine.df)
        self.analyzer = IncidentAnalyzer(model=model)

    def investigate(
        self,
        query: str,
        top_k: int = 10
    ):

        results = self.engine.search_hybrid(
            query,
            top_k=top_k
        )

        context = self.context_builder.build_context(
            results
        )

        patterns = extract_patterns(context)

        analysis = self.analyzer.analyze(
            query, context, patterns
        )

        return analysis