
from incidentiq.reasoning.models import InvestigationResult


def _citation_rate(items, valid_evidence_ids):
    """Fraction of items with at least one valid evidence ID."""
    if not items:
        return 1.0

    cited = sum(
        1
        for item in items
        if any(
            evidence_id in valid_evidence_ids
            for evidence_id in item.evidence_ids
        )
    )

    return cited / len(items)


def _completion_consistency(result: InvestigationResult) -> bool:
    """An incomplete investigation must provide next steps."""
    if not result.analysis.investigation_complete:
        return len(result.analysis.next_steps) > 0

    # Remaining unknowns are allowed when the investigation is complete.
    return True


def _tool_efficiency(result: InvestigationResult):
    """
    Unique tool-retrieved evidence IDs per attempted tool call.

    This measures raw evidence yield, not net-new evidence.
    """
    tool_call_count = len(result.tool_calls)

    if tool_call_count == 0:
        return None

    unique_ids = {
        item["doc_id"]
        for item in result.tool_evidence
    }

    return len(unique_ids) / tool_call_count


def _validate_citations(result: InvestigationResult):
    """
    Validate citation provenance and identify missing citations.

    This checks whether cited IDs were retrieved. It does not
    determine whether the evidence semantically supports a claim.
    """
    valid_evidence_ids = {
        item["doc_id"]
        for item in result.evidence
    }

    valid_evidence_ids.update(
        item["doc_id"]
        for item in result.tool_evidence
    )

    invalid_references = []
    missing_citations = []

    sections = {
        "observations": result.analysis.observations,
        "hypotheses": result.analysis.hypotheses,
        "next_steps": result.analysis.next_steps,
    }

    for section_name, items in sections.items():
        for index, item in enumerate(items):
            if not item.evidence_ids:
                missing_citations.append({
                    "section": section_name,
                    "item_index": index,
                    "reason": "No evidence IDs were provided.",
                })
                continue

            for evidence_id in item.evidence_ids:
                if evidence_id not in valid_evidence_ids:
                    invalid_references.append({
                        "section": section_name,
                        "item_index": index,
                        "evidence_id": evidence_id,
                        "reason": "Evidence ID was not retrieved.",
                    })

    missing_counts = {
        section: sum(
            1
            for item in missing_citations
            if item["section"] == section
        )
        for section in sections
    }

    invalid_reference_counts = {
        section: sum(
            1
            for item in invalid_references
            if item["section"] == section
        )
        for section in sections
    }

    is_grounded = (
        not invalid_references
        and not missing_citations
    )

    return {
        "valid_evidence_ids": valid_evidence_ids,
        "invalid_references": invalid_references,
        "missing_citations": missing_citations,
        "missing_citation_counts": missing_counts,
        "invalid_reference_counts": invalid_reference_counts,
        "is_grounded": is_grounded,
    }


def evaluate_agent(result: InvestigationResult) -> dict:
    """Evaluate structural and citation properties of an investigation."""
    citation_report = _validate_citations(result)
    valid_evidence_ids = citation_report["valid_evidence_ids"]
    analysis = result.analysis

    tool_efficiency = _tool_efficiency(result)

    return {
        "observation_citation_rate": round(
            _citation_rate(
                analysis.observations, valid_evidence_ids
            ),
            3,
        ),
        "hypothesis_citation_rate": round(
            _citation_rate(
                analysis.hypotheses, valid_evidence_ids
            ),
            3,
        ),
        "next_step_citation_rate": round(
            _citation_rate(
                analysis.next_steps, valid_evidence_ids
            ),
            3,
        ),
        "completion_consistent": _completion_consistency(result),
        "tool_efficiency": (
            round(tool_efficiency, 3)
            if tool_efficiency is not None
            else None
        ),
        "grounded": citation_report["is_grounded"],
        "missing_citation_counts": citation_report[
            "missing_citation_counts"
        ],
        "invalid_reference_counts": citation_report[
            "invalid_reference_counts"
        ],
        "missing_citations": citation_report["missing_citations"],
        "citation_violations": citation_report["invalid_references"],
    }
