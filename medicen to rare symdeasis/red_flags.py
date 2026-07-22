from typing import List
from models import PatientState, RedFlag, SymptomEvent

# A simple rule-based mock for red flags
RED_FLAG_RULES = [
    {
        "trigger": "sudden severe headache",
        "keywords": ["thunderclap", "worst headache of my life", "sudden severe headache"],
        "severity": "emergent",
        "message": "Immediate evaluation required to rule out subarachnoid hemorrhage or other acute intracranial pathology. Please seek emergency care."
    },
    {
        "trigger": "chest pain with syncope",
        "keywords": ["chest pain and fainting", "syncope after chest pain", "fainted with chest pain"],
        "severity": "emergent",
        "message": "Chest pain associated with fainting is a cardiac red flag requiring immediate emergency evaluation."
    },
    {
        "trigger": "anaphylaxis signs",
        "keywords": ["throat swelling", "can't breathe", "anaphylaxis"],
        "severity": "emergent",
        "message": "Potential signs of anaphylaxis. Administer epinephrine if available and seek immediate emergency care."
    }
]

def check_red_flags(state: PatientState, current_turn: int) -> List[RedFlag]:
    """
    Independent middleware to check for urgent/emergent conditions based on the current timeline.
    """
    triggered_flags = []
    
    # Simple keyword-based checking on recent symptoms
    
    for rule in RED_FLAG_RULES:
        matched_ids = []
        rule_triggered = False
        
        for kw in rule["keywords"]:
            for sym in state.timeline:
                # We check the raw text for safety
                if kw in sym.raw_text.lower():
                    matched_ids.append(sym.id)
                    rule_triggered = True
                    
        if rule_triggered:
            # Check if this rule is already active to avoid duplicates
            if not any(f.trigger == rule["trigger"] for f in state.red_flags_active):
                triggered_flags.append(RedFlag(
                    trigger=rule["trigger"],
                    matched_symptom_ids=list(set(matched_ids)),
                    severity=rule["severity"], # type: ignore
                    message=rule["message"],
                    triggered_turn=current_turn
                ))
                
    return triggered_flags
