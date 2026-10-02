from pydantic import BaseModel, Field

class Observation(BaseModel):
    statement: str
    evidence_ids: list[int] = Field(default_factory=list)

class Hypothesis(BaseModel):
    statement: str
    evidence_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class InvestigationStep(BaseModel):
    action: str
    query: str | None = None
    reason: str
    evidence_ids: list[int] = Field(default_factory=list)

class IncidentAnalysis(BaseModel):
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list, description=("Important questions that remain unanswerd. An unknown does not necessarily mean the investigation should continue if the available log evidence cannot reasonbly answer it."))
    next_steps: list[InvestigationStep] = Field(default_factory=list, description="Concrete investigation actions that can be performed using the available evidence or search_tool. These should only be included when further investigation could meaningfully reduce an important unknown.")
    investigation_complete: bool = Field(description="True when the available evidence is sufficient for the current investigation, including when remaning unkowns cannot reasonable be resolved using the available logs. False when an important remaining unkown can be meaningfully investigated using seach_logs.")

class AnalyzerResult(BaseModel):
    analysis: IncidentAnalysis
    tool_evidence: list[dict] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)

class GroundingReport(BaseModel):
    is_grounded: bool
    retrieved_evidence_count: int
    violations: list[dict] = Field(default_factory=list)

class InvestigationState(BaseModel):
    query: str
    evidence: list[dict] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    next_steps: list[InvestigationStep] = Field(default_factory=list)
    iteration: int = 0

class InvestigationResult(BaseModel):
    query: str
    analysis: IncidentAnalysis
    evidence: list[dict] = Field(default_factory=list)
    tool_evidence: list[dict] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    patterns: list[dict] = Field(default_factory=list)
    grounding: GroundingReport