import os

from google import genai

from incidentiq.reasoning.prompt import (
    build_reasoning_prompt
)

from incidentiq.reasoning.models import(
    IncidentAnalysis,
    Observation,
    Hypothesis
)

class IncidentAnalyzer:

    def __init__(self, model="gemini-3.6-flash"):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = model

    def analyze(
        self,
        query: str,
        context: dict,
        patterns: list[dict],
    ) -> IncidentAnalysis:

        prompt = build_reasoning_prompt(
            query=query,
            context=context,
            patterns=patterns
        )

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            response_format={
                "type":"text",
                "mime_type":"application/json",
                "schema": IncidentAnalysis.model_json_schema()
            }
        )


        return IncidentAnalysis.model_validate_json(
            interaction.output_text
        )