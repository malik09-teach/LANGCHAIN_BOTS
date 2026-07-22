from typing import List, Dict, Any
from models import DiseaseHypothesis, PatientState

def annotate_confidence(hypotheses: List[DiseaseHypothesis], state: PatientState) -> Dict[str, Any]:
    """
    Scores evidence sufficiency and identifies questions to ask.
    """
    suggested_questions = []
    
    for h in hypotheses:
        # If there are unmatched expected terms, ask about them
        if h.posterior > 0.01: # Only ask if the hypothesis is somewhat likely
            for unmatched in h.unmatched_expected_terms:
                if unmatched == "HP:0001250":
                    suggested_questions.append("Has the patient ever experienced any seizures?")
                elif unmatched == "HP:0002829":
                    suggested_questions.append("Is there any history of joint pain or arthralgia?")
                elif unmatched == "HP:0002076":
                    suggested_questions.append("Has the patient experienced migraines with aura?")
                
    # Deduplicate questions
    suggested_questions = list(set(suggested_questions))
    
    return {
        "hypotheses": hypotheses,
        "suggested_questions": suggested_questions
    }
