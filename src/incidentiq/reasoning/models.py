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
    reason: str
    evidence_ids: list[int] = Field(default_factory=list)

class IncidentAnalysis(BaseModel):
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    next_steps: list[InvestigationStep] = Field(default_factory=list)

class InvestigationResult(BaseModel):
    query: str
    analysis: IncidentAnalysis
    evidence: list[dict] = Field(default_factory=list)
    patterns: list[dict] = Field(default_factory=list)