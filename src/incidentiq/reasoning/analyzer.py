import json
import time

from incidentiq.config import GEMINI_API_KEY
from google import genai

from incidentiq.reasoning.prompt import (
    build_reasoning_prompt
)

from incidentiq.reasoning.models import (
    IncidentAnalysis,
    AnalyzerResult,
    Observation,
    Hypothesis,
    InvestigationState
)


class IncidentAnalyzer:

    def __init__(
        self,
        model="gemini-3.5-flash-lite",
        search_tool=None
    ):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = model
        self.search_tool = search_tool

    def analyze(
        self,
        query: str,
        context: dict,
        patterns: list[dict],
        state: InvestigationState | None = None
    ) -> AnalyzerResult:

        start = time.perf_counter()

        prompt = build_reasoning_prompt(
            query=query,
            context=context,
            patterns=patterns
        )

        if state is not None:
            state_context = {
                "iteration": state.iteration,
                "previous_observations": [
                    observation.model_dump()
                    for observation in state.observations
                ],
                "previous_hypotheses": [
                    hypothesis.model_dump()
                    for hypothesis in state.hypotheses
                ],
                "unknowns": state.unknowns,
                "previous_tool_calls": state.tool_calls,
                "evidence": state.evidence
            }

            prompt += (
                "\n\nCURRENT INVESTIGATION STATE:\n"
                + json.dumps(
                    state_context,
                    default=str,
                    indent=2
                )
                + "\n\n"
                "Continue the investigation from this state. "
                "Do not repeat work unnecessarily. "
                "Use search_logs if additional evidence is needed. "
                "Set investigation_complete=true only when the available "
                "evidence is sufficient for the current investigation."
            )

        prompt_time = time.perf_counter()

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            tools=self.get_tools(),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": IncidentAnalysis.model_json_schema()
            }
        )

        api_time = time.perf_counter()

        # Evidence and tool-call tracking for this reasoning cycle
        tool_evidence = []
        tool_calls = []

        # Prevent duplicate evidence from being added
        seen_evidence_ids = set()

        # Prevent the exact same query from being executed twice
        seen_queries = set()

        # Safety budget for tool calls
        MAX_TOOL_CALLS = 10

        while True:

            # Every attempted tool call counts toward the budget.
            if len(tool_calls) >= MAX_TOOL_CALLS:
                raise RuntimeError(
                    f"Maximum tool calls ({MAX_TOOL_CALLS}) reached "
                    "without the model producing a final analysis."
                )

            function_call = next(
                (
                    step
                    for step in interaction.steps
                    if step.type == "function_call"
                ),
                None
            )

            # No tool call means the model has produced its final analysis.
            if function_call is None:

                analysis = IncidentAnalysis.model_validate_json(
                    interaction.output_text
                )

                print(
                    "Total tool evidence : ",
                    len(tool_evidence)
                )

                print(
                    "Tool evidence IDs: ",
                    [
                        item["doc_id"]
                        for item in tool_evidence
                    ]
                )

                parse_time = time.perf_counter()

                print(
                    f"Prompt: {prompt_time - start:.3f}s | "
                    f"Gemini API: {api_time - prompt_time:.3f}s | "
                    f"Parsing: {parse_time - api_time:.3f}s"
                )

                return AnalyzerResult(
                    analysis=analysis,
                    tool_evidence=tool_evidence,
                    tool_calls=tool_calls
                )

            print(
                "FUNCTION NAME:",
                repr(function_call.name)
            )

            print(
                "FUNCTION NAME TYPE:",
                type(function_call.name)
            )

            print(
                "EXPECTED:",
                repr("search_logs")
            )

            print(
                "EQUAL:",
                function_call.name == "search_logs"
            )

            if function_call.name != "search_logs":
                raise ValueError(
                    f"Unknown tool requested: "
                    f"{function_call.name}"
                )

            arguments = function_call.arguments

            search_query = arguments["query"]

            # Count every attempted tool call.
            # This prevents duplicate-query loops from bypassing
            # the MAX_TOOL_CALLS safety budget.
            tool_calls.append(search_query)

            print(
                f"Tool call : {function_call.name} "
                f"with arguments: {arguments}"
            )

            # ---------------------------------------------------------
            # DUPLICATE QUERY DETECTION
            # ---------------------------------------------------------

            if search_query in seen_queries:

                print(
                    f"Skipping duplicate search query: "
                    f"{search_query}"
                )

                interaction = self.client.interactions.create(
                    model=self.model,
                    previous_interaction_id=interaction.id,
                    tools=self.get_tools(),
                    input=[
                        {
                            "type": "function_result",
                            "name": function_call.name,
                            "call_id": function_call.id,
                            "result": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "error": "duplicate_query",
                                            "message": (
                                                "This exact search query "
                                                "has already been executed. "
                                                "Use a different query if "
                                                "additional evidence is "
                                                "needed."
                                            )
                                        }
                                    )
                                }
                            ]
                        }
                    ],
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": (
                            IncidentAnalysis
                            .model_json_schema()
                        )
                    }
                )

                api_time = time.perf_counter()

                continue

            # First time seeing this query
            seen_queries.add(search_query)

            # ---------------------------------------------------------
            # EXECUTE SEARCH
            # ---------------------------------------------------------

            results = self.search_tool.search(
                query=search_query,
                top_k=arguments.get("top_k", 10)
            )

            # ---------------------------------------------------------
            # DEDUPLICATE EVIDENCE
            # ---------------------------------------------------------

            for result in results:

                doc_id = result["doc_id"]

                if doc_id not in seen_evidence_ids:

                    tool_evidence.append(result)
                    seen_evidence_ids.add(doc_id)

            print(
                f"Tool returned {len(results)} results."
            )

            # ---------------------------------------------------------
            # RETURN TOOL RESULT TO GEMINI
            # ---------------------------------------------------------

            interaction = self.client.interactions.create(
                model=self.model,
                previous_interaction_id=interaction.id,
                tools=self.get_tools(),
                input=[
                    {
                        "type": "function_result",
                        "name": function_call.name,
                        "call_id": function_call.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    results,
                                    default=str
                                )
                            }
                        ]
                    }
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": (
                        IncidentAnalysis
                        .model_json_schema()
                    )
                }
            )

            api_time = time.perf_counter()

    def get_tools(self):

        return [
            {
                "type": "function",
                "name": "search_logs",
                "description": (
                    "Search the incident log corpus for additional evidence. "
                    "Use this when the supplied evidence is insufficient "
                    "to investigate the incident."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {

                        "query": {
                            "type": "string",
                            "description": (
                                "A focused search query describing the "
                                "logs you want to investigate."
                            )
                        },

                        "top_k": {
                            "type": "integer",
                            "description": (
                                "Maximum number of log events "
                                "to return."
                            ),
                            "minimum": 1,
                            "maximum": 10
                        }
                    },

                    "required": ["query"]
                }
            }
        ]


if __name__ == "__main__":

    analyzer = IncidentAnalyzer()

    interaction = analyzer.client.interactions.create(
        model=analyzer.model,
        input=(
            "Investigate the incident 'node card failure'. "
            "You have some evidence, but if you need additional "
            "evidence, use the search_logs tool."
        ),
        tools=analyzer.get_tools(),
        generation_config={
            "tool_choice": {
                "allowed_tools": {
                    "mode": "any",
                    "tools": ["search_logs"]
                }
            }
        }
    )

    for step in interaction.steps:

        print(
            "TYPE:",
            step.type
        )

        print(
            "NAME:",
            getattr(step, "name", None)
        )

        print(
            "ARGS:",
            getattr(step, "arguments", None)
        )

        print(
            "CALL ID:",
            getattr(step, "id", None)
        )