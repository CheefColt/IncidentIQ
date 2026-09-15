import time
from incidentiq.context.builder import ContextBuilder
from incidentiq.context.patterns import extract_patterns
from incidentiq.reasoning.analyzer import IncidentAnalyzer
from incidentiq.search import SearchEngine
from incidentiq.reasoning.models import InvestigationResult
from incidentiq.tools.search import LogSearchTool

class IncidentIQ:

    def __init__(
        self,
        data_path: str,
        model: str = "gemini-3.5-flash-lite"
    ):

        self.engine = SearchEngine(data_path)
        self.search_tool = LogSearchTool(self.engine)

        self.context_builder = ContextBuilder(self.engine.df)
        self.analyzer = IncidentAnalyzer(model=model, search_tool=self.search_tool)

    def investigate(
        self,
        query: str,
        top_k: int = 10
    )-> InvestigationResult:

        start = time.perf_counter()

        results = self.engine.search_hybrid(
            query,
            top_k=top_k
        )

        retrieval_time = time.perf_counter()

        context = self.context_builder.build_context(
            results
        )

        context_time = time.perf_counter()

        patterns = extract_patterns(context)

        patterns_time = time.perf_counter()

        analysis = self.analyzer.analyze(
            query, context, patterns
        )

        reasoning_time = time.perf_counter()

        print(
        f"Retrieval: {retrieval_time - start:.3f}s | "
        f"Context: {context_time - retrieval_time:.3f}s | "
        f"Patterns: {patterns_time - context_time:.3f}s | "
        f"Reasoning: {reasoning_time - patterns_time:.3f}s | "
        f"Total: {reasoning_time - start:.3f}s")

        return InvestigationResult(
            query=query,
            analysis=analysis,
            evidence=context["evidence"],
            patterns=patterns
        )