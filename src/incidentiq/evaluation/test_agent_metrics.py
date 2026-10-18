# pyrefly: ignore [missing-import]
from incidentiq.evaluation.agent_metrics import evaluate_agent
from incidentiq.reasoning.models import (
    IncidentAnalysis,
    InvestigationResult,
    GroundingReport,
    Observation,
    Hypothesis,
    InvestigationStep
)

def make_result(
    *,
    investigation_complete = True,
    observations = None,
    hypotheses = None,
    next_steps = None,
    unknowns = None,
    evidence = None,
    tool_evidence = None,
    tool_calls = None,
    grounded = True
):
    """ Build a small InvestigationResult for deterministic tests."""

    analysis = IncidentAnalysis(
        summary = "Test Investigation",
        observations=observations or [],
        hypotheses=hypotheses or[],
        unknowns=unknowns or [],
        next_steps=next_steps or [],
        investigation_complete=investigation_complete
    )

    evidence = evidence or [
        {
            "doc_id": 1,
            "timestamp": "2005-06-01 10:00:00",
            "node": "R01-M0-N0",
            "severity": "WARNING",
            "message": "Machine check timeout",
        }
    ]

    return InvestigationResult(
        query = "machine check timeout",
        analysis=analysis,
        evidence=evidence,
        tool_evidence=tool_evidence or [],
        tool_calls=tool_calls or [],
        patterns = [],
        grounding=GroundingReport(
            is_grounded=grounded,
            retrieved_evidence_count=len(evidence),
            violations=[],
        )
    )


def test_valid_investigation_passes():
    result = make_result(
        observations=[
            Observation(
                statement="A machine check timeout was recorded.",
                evidence_ids=[1],
            )
        ],
        hypotheses=[
            Hypothesis(
                statement="A hardware issue may be involved.",
                evidence_ids=[1],
                confidence=0.6
            )
        ],
        next_steps=[
            InvestigationStep(
                action="Search for additional machine-check events.",
                query="machine check",
                reason="Determine whether the event recurs.",
                evidence_ids=[1],
            )
        ],
        investigation_complete=False
    )

    metrics = evaluate_agent(result)

    assert metrics["observation_citation_rate"] == 1.0
    assert metrics["hypothesis_citation_rate"] == 1.0
    assert metrics["next_step_citation_rate"] == 1.0
    assert metrics["completion_consistent"] is True
    assert metrics["grounded"] is True



def test_nonexistent_observation_evidence_is_detected():
    result = make_result(
        observations=[
            Observation(
                statement="An unsupported event occurred.",
                evidence_ids=[999999],
            )
        ]
    )

    metrics = evaluate_agent(result)

    assert metrics["observation_citation_rate"] == 0.0
    assert metrics["grounded"] is False
    assert len(metrics["citation_violations"]) == 1

    violation = metrics["citation_violations"][0]

    assert violation["section"] == "observations"
    assert violation["evidence_id"] == 999999
    assert violation["reason"] == "Evidence ID was not retrieved."


def test_incomplete_investigation_without_next_steps_is_detected():

    result = make_result(
        investigation_complete=False,
        next_steps=[]
    )

    metrics = evaluate_agent(result)

    assert metrics["completion_consistent"] is False


def test_missing_observation_citation_is_detected():
    result = make_result(
        observations=[
            Observation(
                statement="An observation without a citation.",
                evidence_ids=[],
            )
        ]
    )

    metrics = evaluate_agent(result)

    assert metrics["observation_citation_rate"] == 0.0
    assert metrics["grounded"] is False

    assert metrics["missing_citation_counts"]["observations"] == 1
    assert metrics["invalid_reference_counts"]["observations"] == 0

    assert len(metrics["missing_citations"]) == 1
    assert metrics["missing_citations"][0]["section"] == "observations"
