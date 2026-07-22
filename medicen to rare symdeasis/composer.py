from typing import List, Dict, Any
from models import PatientState, RedFlag, DiseaseHypothesis

def build_response(red_flags: List[RedFlag], scored_result: Dict[str, Any], open_questions: List[str]) -> str:
    """
    Assembles the final output structure.
    """
    hypotheses: List[DiseaseHypothesis] = scored_result["hypotheses"]
    suggested_questions: List[str] = scored_result["suggested_questions"]
    
    response = []
    
    # 1. Red Flags
    if red_flags:
        response.append("### 🚨 URGENT CLINICAL ALERTS 🚨")
        for flag in red_flags:
            response.append(f"- **{flag.severity.upper()}**: {flag.message}")
        response.append("\n---\n")
            
    # 2. Ranked Differential
    response.append("### Ranked Differential Hypothesis")
    if not hypotheses:
        response.append("Insufficient data to generate a rare disease differential.")
    else:
        for idx, h in enumerate(hypotheses[:5]): # Top 5
            response.append(f"**{idx+1}. {h.name}**")
            prob_pct = h.posterior * 100
            response.append(f"   - **Confidence**: {prob_pct:.1f}%")
            if h.matched_hpo_terms:
                response.append(f"   - **Matched Phenotypes**: {', '.join(h.matched_hpo_terms)}")
            if h.citations:
                cites = ", ".join([f"[{c.source}]({c.url})" for c in h.citations])
                response.append(f"   - **Citations**: {cites}")
            response.append("")
            
    # 3. Suggested Questions
    all_questions = open_questions + suggested_questions
    if all_questions:
        response.append("### Suggested Next Steps / Clarifications")
        for q in all_questions:
            response.append(f"- {q}")
            
    # 4. Disclaimer
    response.append("\n---\n")
    response.append("*Disclaimer: This is a decision-support aid, not a diagnostic instrument. It does not replace clinical judgment or genetic testing.*")
    
    return "\n".join(response)
