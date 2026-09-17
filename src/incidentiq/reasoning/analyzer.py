import json
import time
from incidentiq.config import GEMINI_API_KEY
from google import genai

from incidentiq.reasoning.prompt import (
    build_reasoning_prompt
)

from incidentiq.reasoning.models import(
    IncidentAnalysis,
    AnalyzerResult,
    Observation,
    Hypothesis
)

class IncidentAnalyzer:

    def __init__(self, model="gemini-3.5-flash-lite",search_tool=None):
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
    ) -> AnalyzerResult:

        start = time.perf_counter()

        prompt = build_reasoning_prompt(
            query=query,
            context=context,
            patterns=patterns
        )  

        prompt_time = time.perf_counter()

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            tools=self.get_tools(),
            response_format={
                "type":"text",
                "mime_type":"application/json",
                "schema": IncidentAnalysis.model_json_schema()
            }
        )

        api_time = time.perf_counter()

        tool_evidence = []

        while True:

            function_call = next(
                (
                    step
                    for step in interaction.steps
                    if step.type == "function_call"
                ),
                None
            )

            if function_call is None:
                analysis = IncidentAnalysis.model_validate_json(
                            interaction.output_text
                        )

                print("Total tool evidence : ", len(tool_evidence))
                print(
                    "Tool evidence IDs: ",
                    [item["doc_id"] for item in tool_evidence]
                )
        
                parse_time = time.perf_counter()
        
                print(
                    f"Prompt: {prompt_time - start:.3f}s | "
                    f"Gemini API: {api_time - prompt_time:.3f}s | "
                    f"Parsing: {parse_time - api_time:.3f}s"
                )

                return AnalyzerResult(
                    analysis=analysis,
                    tool_evidence=tool_evidence
                )

            print("FUNCTION NAME:", repr(function_call.name))
            print("FUNCTION NAME TYPE:", type(function_call.name))
            print("EXPECTED:", repr("search_logs"))
            print("EQUAL:", function_call.name == "search_logs")

            if function_call.name != "search_logs":
                raise ValueError(
                    f"Unkown tool requested: {function_call.name}"
                )

            arguments = function_call.arguments

            print(
                f"Tool call : {function_call.name} "
                f"with arguments: {arguments}"
            )

            results = self.search_tool.search(
                query=arguments["query"],
                top_k=arguments.get("top_k", 10)
            )

            tool_evidence.extend(results)

            print(f"Tool returned {len(results)} results.")

            interaction = self.client.interactions.create(
                model = self.model,
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
                    "schema": IncidentAnalysis.model_json_schema()
                }
            )

            api_time = time.perf_counter()

    def get_tools(self):

        return [
            {
                "type": "function",
                "name": "search_logs",
                "description": (
                    "Search the incident log corpus for additional evidence."
                    "Use this when the supplied evidence is insufficient to "
                    "investivate the incident."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A focused search query describing the logs "
                                "you want to investigate."
                            )
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of logs events to return.",
                            "minimum": 1,
                            "maximum": 10
                        },
                    },
                    "required": ["query"],
                }
            }
        ]


if __name__=="__main__":
    analyzer = IncidentAnalyzer()

    interaction = analyzer.client.interactions.create(
        model = analyzer.model,
        input=(
            "Investigate the incident 'node card failure'. "
            "You have some evidence, but if you need additional evidence, "
            "use the search_logs tool."
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
        print("TYPE:", step.type)
        print("NAME:", getattr(step, "name", None))
        print("ARGS:", getattr(step, "arguments", None))
        print("CALL ID:", getattr(step, "id", None))