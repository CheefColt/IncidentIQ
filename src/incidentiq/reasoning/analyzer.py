from incidentiq.reasoning.models import(
    IncidentAnalysis,
    Observation,
    Hypothesis
)

class IncidentAnalyzer:

    def analyze(
        self,
        query: str,
        context: dict,
        patterns: list[dict],
    ) -> IncidentAnalysis:

        observations = []
        hypotheses = []

        # Build observations
        for pattern in patterns:

            if pattern["type"] == "repeated_message":

                observations.append(
                    Observation(
                        statement=(
                            f"The message "
                            f"'{pattern['message']}' "
                            f"appears in "
                            f"{pattern['occurrences']} retrieved events."
                        ),
                        evidence_ids=pattern["evidence_ids"],
                    )
                )

        # Build hypothesis
        if len(observations) >= 2:

            evidence_ids = []

            for observation in observations:
                evidence_ids.extend(observation.evidence_ids)

            hypotheses.append(
                Hypothesis(
                    statement=(
                        "The repeated node-card functionality and "
                        "assembly-information errors may indicate a "
                        "broader node-card hardware or configuration issue."
                    ),
                    evidence_ids=evidence_ids,
                    confidence=0.65,
                )
            )

        unknowns = [
            "The available evidence does not establish the underlying root cause.",
            "The retrieved events span multiple months and cannot be assumed to represent one continuous incident.",
            "The evidence does not establish whether the affected nodes share a common hardware component or configuration.",
        ]

        return IncidentAnalysis(
            summary=f"Investigation results for: {query}",
            observations=observations,
            hypotheses=hypotheses,
            unknowns=unknowns,
        )