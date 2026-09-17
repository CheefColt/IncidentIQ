def validate_evidence_citations(
        analysis,
        initial_evidence: list[dict],
        tool_evidence: list[dict]
) -> dict:

    retrieved_ids = {
        item["doc_id"]
        for item in initial_evidence + tool_evidence
    }

    violations = []

    for obs in analysis.observations:
        invalid_ids = [
            evidence_id
            for evidence_id in obs.evidence_ids
            if evidence_id not in retrieved_ids
        ]

        if invalid_ids:
            violations.append({
                "type": "observation",
                "statement": obs.statement,
                "invalid_evidence_ids": invalid_ids
            })

    for hypothesis in analysis.hypotheses:
        invalid_ids = [
                    evidence_id
                    for evidence_id in hypothesis.evidence_ids
                    if evidence_id not in retrieved_ids
                ]
        
        if invalid_ids:
            violations.append({
                "type": "hypothesis",
                "statement": hypothesis.statement,
                "invalid_evidence_ids": invalid_ids
            })

    for step in analysis.next_steps:
        invalid_ids = [
             evidence_id
             for evidence_id in step.evidence_ids
             if evidence_id not in retrieved_ids
         ]
                
        if invalid_ids:
            violations.append({
                "type": "next_step",
                "statement": step.action,
                "invalid_evidence_ids": invalid_ids
            })

    return {
        "is_grounded": len(violations) == 0,
        "retrieved_evidence_count": len(retrieved_ids),
        "violations": violations
    }