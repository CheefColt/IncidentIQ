from pydantic import BaseModel, Field

class Observation(BaseModel):
    statement: str
    evidence_ids: list[int] = Field(default_factory=list)

class Hypothesis(BaseModel):
    statement: str
    evidence_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class IncidentAnalysis(BaseModel):
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)