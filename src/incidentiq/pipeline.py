import time

from incidentiq.context.builder import ContextBuilder
from incidentiq.context.patterns import extract_patterns
from incidentiq.reasoning.analyzer import IncidentAnalyzer
from incidentiq.search import SearchEngine
from incidentiq.reasoning.models import (
    InvestigationResult,
    GroundingReport,
    InvestigationState
)
from incidentiq.tools.search import LogSearchTool
from incidentiq.evaluation.grounding import validate_evidence_citations


class IncidentIQ:

    def __init__(
        self,
        data_path: str,
        model: str = "gemini-3.5-flash-lite"
    ):

        self.engine = SearchEngine(data_path)
        self.search_tool = LogSearchTool(self.engine)

        self.context_builder = ContextBuilder(
            self.engine.df
        )

        self.analyzer = IncidentAnalyzer(
            model=model,
            search_tool=self.search_tool
        )

    def investigate(
        self,
        query: str,
        top_k: int = 10
    ) -> InvestigationResult:

        start = time.perf_counter()

        # --------------------------------------------------
        # 1. Initial retrieval
        # --------------------------------------------------

        results = self.engine.search_hybrid(
            query,
            top_k=top_k
        )

        retrieval_time = time.perf_counter()

        # --------------------------------------------------
        # 2. Build initial context
        # --------------------------------------------------

        context = self.context_builder.build_context(
            results
        )

        initial_context = context

        context_time = time.perf_counter()

        # --------------------------------------------------
        # 3. Initialize investigation state
        # --------------------------------------------------

        state = InvestigationState(
            query=query,
            evidence=context["evidence"],
            iteration=0
        )

        # --------------------------------------------------
        # 4. Investigation loop
        # --------------------------------------------------

        MAX_ITERATIONS = 3

        final_analysis = None

        all_tool_evidence = []
        all_tool_calls = []

        # Keep track of every evidence ID that has already
        # entered the investigation.
        seen_evidence_ids = {
            item["doc_id"]
            for item in state.evidence
        }

        patterns = []

        while True:

            print(
                f"\n------ Investigation Iteration "
                f"{state.iteration + 1} ------"
            )

            analyzer_result = self.analyzer.analyze(
                query=query,
                context=context,
                patterns=patterns,
                state=state
            )

            final_analysis = analyzer_result.analysis

            # --------------------------------------------------
            # Accumulate tool calls
            # --------------------------------------------------

            all_tool_calls.extend(
                analyzer_result.tool_calls
            )

            state.tool_calls.extend(
                analyzer_result.tool_calls
            )

            # --------------------------------------------------
            # Accumulate tool evidence
            #
            # Only add evidence that has not already appeared
            # anywhere in this investigation.
            # --------------------------------------------------

            new_tool_evidence = []

            for evidence in analyzer_result.tool_evidence:

                doc_id = evidence["doc_id"]

                if doc_id not in seen_evidence_ids:

                    seen_evidence_ids.add(doc_id)

                    new_tool_evidence.append(
                        evidence
                    )

                    all_tool_evidence.append(
                        evidence
                    )

                    state.evidence.append(
                        evidence
                    )

            print(
                f"New evidence this iteration: "
                f"{len(new_tool_evidence)}"
            )

            print(
                f"Total unique investigation evidence: "
                f"{len(state.evidence)}"
            )

            # --------------------------------------------------
            # Update investigation state
            # --------------------------------------------------

            state.observations = (
                analyzer_result.analysis.observations
            )

            state.hypotheses = (
                analyzer_result.analysis.hypotheses
            )

            state.unknowns = (
                analyzer_result.analysis.unknowns
            )

            state.next_steps = (
                analyzer_result.analysis.next_steps
            )

            state.iteration += 1

            print(
                "Iteration complete:",
                state.iteration
            )

            print(
                "Investigation complete:",
                final_analysis.investigation_complete
            )

            # --------------------------------------------------
            # Stop if investigation is complete
            # --------------------------------------------------

            if final_analysis.investigation_complete:
                break

            # --------------------------------------------------
            # Stop if iteration limit is reached
            # --------------------------------------------------

            if state.iteration >= MAX_ITERATIONS:

                print(
                    "Maximum investigation iterations reached."
                )

                break

            # --------------------------------------------------
            # Rebuild context for next iteration
            # --------------------------------------------------

            context = self.context_builder.build_context(
                state.evidence
            )

        reasoning_time = time.perf_counter()

        # --------------------------------------------------
        # 5. Grounding
        # --------------------------------------------------

        grounding = validate_evidence_citations(
            analysis=final_analysis,
            initial_evidence=initial_context["evidence"],
            tool_evidence=all_tool_evidence
        )

        # --------------------------------------------------
        # 6. Timing
        # --------------------------------------------------

        print(
            f"Retrieval: {retrieval_time - start:.3f}s | "
            f"Context: {context_time - retrieval_time:.3f}s | "
            f"Reasoning: {reasoning_time - context_time:.3f}s | "
            f"Total: {reasoning_time - start:.3f}s"
        )

        # --------------------------------------------------
        # 7. Return InvestigationResult
        # --------------------------------------------------

        return InvestigationResult(
            query=query,
            analysis=final_analysis,
            evidence=state.evidence,
            tool_evidence=all_tool_evidence,
            tool_calls=all_tool_calls,
            patterns=patterns,
            grounding=GroundingReport(**grounding)
        )