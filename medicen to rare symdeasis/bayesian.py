from typing import List
from models import DiseaseHypothesis, PatientState

def rerank(hypotheses: List[DiseaseHypothesis], state: PatientState, current_turn: int) -> List[DiseaseHypothesis]:
    """
    Deterministic Bayesian re-ranking tool.
    """
    for h in hypotheses:
        if not h.stale:
            continue
            
        # Simplified Bayesian update
        # P(H|E) = P(E|H) * P(H) / P(E)
        
        new_posterior = h.prior
        
        # Mock likelihood ratios (LR) based on presence of symptoms
        hpo_terms = [sym.hpo_id for sym in state.timeline if sym.hpo_id]
        
        for hpo in hpo_terms:
            if hpo in h.matched_hpo_terms:
                # Symptom present, increases probability
                # In a real system, LR comes from structured frequency data
                likelihood_ratio = 5.0 
                
                # Update odds
                odds = new_posterior / (1.0 - new_posterior)
                new_odds = odds * likelihood_ratio
                new_posterior = new_odds / (1.0 + new_odds)
                
        h.posterior = new_posterior
        h.stale = False # Marked as freshly scored
        
    # Sort hypotheses by posterior descending
    hypotheses.sort(key=lambda x: x.posterior, reverse=True)
    
    return hypotheses
