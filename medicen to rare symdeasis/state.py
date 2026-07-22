from typing import List, Tuple
from models import PatientState, SymptomEvent

def apply_diffs(state: PatientState, new_events: List[SymptomEvent], contradictions: List[str], current_turn: int) -> PatientState:
    """
    Applies parser diffs to PatientState.
    """
    state.last_updated_turn = current_turn
    
    for event in new_events:
        # Check if we are updating an existing symptom or adding a new one
        # For this prototype, we just append to the timeline
        state.timeline.append(event)
        
    # In a real system, we'd mark hypotheses as stale if their supporting symptoms change
    if contradictions:
        state.open_questions.extend(contradictions)
        
    return state
