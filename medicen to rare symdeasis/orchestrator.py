from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from models import PatientState, DiseaseHypothesis, SymptomEvent
from parser import TemporalSymptomParser
from state import apply_diffs
from red_flags import check_red_flags
from hypothesis import update_candidates
from rag import retrieve_evidence
from bayesian import rerank
from confidence import annotate_confidence
from composer import build_response

# Define the overall state for the LangGraph
class AppState(TypedDict):
    user_text: str
    patient_state: PatientState
    current_turn: int
    hypotheses: List[DiseaseHypothesis]
    response: str

def process_turn_node(state: AppState) -> AppState:
    user_text = state["user_text"]
    p_state = state["patient_state"]
    turn = state["current_turn"]
    active_hypotheses = state["hypotheses"]
    
    parser = TemporalSymptomParser()
    new_events, contradictions = parser.parse(user_text, turn)
    
    p_state = apply_diffs(p_state, new_events, contradictions, turn)
    
    # Red flags
    red_flags = check_red_flags(p_state, turn)
    p_state.red_flags_active = red_flags
    
    # Pipeline
    candidates = update_candidates(p_state, turn, active_hypotheses)
    candidates = retrieve_evidence(candidates, p_state, turn)
    scored = rerank(candidates, p_state, turn)
    confidence_result = annotate_confidence(scored, p_state)
    
    final_response = build_response(red_flags, confidence_result, p_state.open_questions)
    
    # Update state
    state["patient_state"] = p_state
    state["hypotheses"] = scored
    state["response"] = final_response
    
    return state

# Setup Graph
workflow = StateGraph(AppState)
workflow.add_node("process_turn", process_turn_node)
workflow.set_entry_point("process_turn")
workflow.add_edge("process_turn", END)
app = workflow.compile()

def handle_turn(user_text: str, patient_state: PatientState, hypotheses: List[DiseaseHypothesis], turn: int) -> str:
    initial_state = AppState(
        user_text=user_text,
        patient_state=patient_state,
        current_turn=turn,
        hypotheses=hypotheses,
        response=""
    )
    
    result = app.invoke(initial_state)
    
    return result["response"]
