from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

class SymptomEvent(BaseModel):
    id: str
    raw_text: str
    symptom_normalized: str
    hpo_id: Optional[str] = None
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Literal["mild", "moderate", "severe", "unknown"] = "unknown"
    status: Literal["active", "resolved", "intermittent", "unknown"] = "unknown"
    associated_triggers: List[str] = Field(default_factory=list)
    turn_index: int
    confidence: float

class RedFlag(BaseModel):
    trigger: str
    matched_symptom_ids: List[str] = Field(default_factory=list)
    severity: Literal["urgent", "emergent"]
    message: str
    triggered_turn: int

class Demographics(BaseModel):
    age_range: Optional[str] = None
    sex_assigned: Optional[str] = None
    ancestry_if_disclosed: Optional[str] = None

class PatientState(BaseModel):
    patient_id: str
    symptom_events: List[SymptomEvent] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    demographics: Demographics = Field(default_factory=Demographics)
    timeline: List[SymptomEvent] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    red_flags_active: List[RedFlag] = Field(default_factory=list)
    last_updated_turn: int = 0

class Citation(BaseModel):
    source: Literal["NORD", "OMIM", "PubMed"]
    source_id: str
    url: str

class EvidenceItem(BaseModel):
    source: Literal["NORD", "OMIM", "PubMed"]
    source_id: str
    excerpt: str
    relevance_score: float
    retrieved_turn: int
    url: str

class DiseaseHypothesis(BaseModel):
    disease_id: str
    name: str
    prior: float
    posterior: float
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)
    matched_hpo_terms: List[str] = Field(default_factory=list)
    unmatched_expected_terms: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    last_scored_turn: int
    stale: bool = False
