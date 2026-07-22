from typing import List
from models import DiseaseHypothesis, PatientState, EvidenceItem, Citation

def retrieve_evidence(hypotheses: List[DiseaseHypothesis], state: PatientState, current_turn: int) -> List[DiseaseHypothesis]:
    """
    RAG Retrieval Loop (Mocked)
    """
    for h in hypotheses:
        if not h.stale:
            continue
            
        # Mock retrieval from a vector store
        if h.disease_id == "MIM:141500":
            h.supporting_evidence = [
                EvidenceItem(
                    source="OMIM",
                    source_id="141500",
                    excerpt="Familial hemiplegic migraine is characterized by recurrent attacks of severe headache.",
                    relevance_score=0.88,
                    retrieved_turn=current_turn,
                    url="https://omim.org/entry/141500"
                )
            ]
            h.citations = [Citation(source="OMIM", source_id="141500", url="https://omim.org/entry/141500")]
            
        if h.disease_id == "NORD:SystemicLupus":
            h.supporting_evidence = [
                EvidenceItem(
                    source="NORD",
                    source_id="SystemicLupus",
                    excerpt="Skin rash is a common presenting symptom of SLE.",
                    relevance_score=0.85,
                    retrieved_turn=current_turn,
                    url="https://rarediseases.org/rare-diseases/systemic-lupus-erythematosus/"
                )
            ]
            h.citations = [Citation(source="NORD", source_id="SystemicLupus", url="https://rarediseases.org/rare-diseases/systemic-lupus-erythematosus/")]
            
    return hypotheses
