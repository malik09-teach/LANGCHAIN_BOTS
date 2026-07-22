import uuid
from typing import List
from models import PatientState, DiseaseHypothesis

def update_candidates(state: PatientState, current_turn: int, active_hypotheses: List[DiseaseHypothesis]) -> List[DiseaseHypothesis]:
    """
    Generates or updates hypotheses based on new symptoms.
    """
    new_hypotheses = active_hypotheses.copy()
    
    # Mock hypothesis generation logic based on HPO terms
    hpo_terms = [sym.hpo_id for sym in state.timeline if sym.hpo_id]
    
    if "HP:0002315" in hpo_terms and not any(h.disease_id == "MIM:141500" for h in active_hypotheses):
        # E.g., Familial Hemiplegic Migraine
        new_hypotheses.append(DiseaseHypothesis(
            disease_id="MIM:141500",
            name="Familial Hemiplegic Migraine",
            prior=0.01,
            posterior=0.01,
            matched_hpo_terms=["HP:0002315"],
            unmatched_expected_terms=["HP:0001250", "HP:0002076"], # Seizures, Migraine with aura
            last_scored_turn=current_turn,
            stale=True # Needs RAG loop to score
        ))
        
    if "HP:0000988" in hpo_terms and not any(h.disease_id == "NORD:SystemicLupus" for h in active_hypotheses):
        new_hypotheses.append(DiseaseHypothesis(
            disease_id="NORD:SystemicLupus",
            name="Systemic Lupus Erythematosus",
            prior=0.005,
            posterior=0.005,
            matched_hpo_terms=["HP:0000988"],
            unmatched_expected_terms=["HP:0002829", "HP:0001369"], # Arthralgia, Arthritis
            last_scored_turn=current_turn,
            stale=True
        ))
        
    # Mark existing hypotheses as stale if timeline updated this turn
    if len(state.timeline) > 0 and state.timeline[-1].turn_index == current_turn:
        for h in new_hypotheses:
            h.stale = True
            
    return new_hypotheses
