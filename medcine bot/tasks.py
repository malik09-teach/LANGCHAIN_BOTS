from crewai import Task
from agents import medical_researcher, formulation_scientist, stability_tester

def create_research_task(formula: str):
    return Task(
        description=f'Search Wikipedia for the medical formula or compound string: {formula}. Identify its standard name, primary uses, side effects, and targeted illnesses.',
        expected_output='A detailed summary including the drug name, chemical formula, mechanism of action, treated diseases, and common side effects.',
        agent=medical_researcher
    )

def create_innovation_task():
    return Task(
        description=(
            'Analyze the pharmacological summary provided by the Researcher. Your goal is to engineer a next-generation breakthrough:\n'
            '1. COMPOUND SUITABILITY: Assess why the current compound is or is not perfectly suitable for its targets.\n'
            '2. NEW FORMULA DISCOVERY: Predict and propose a specific new, theoretical chemical formula or structural modification (e.g., adding a functional group, creating a hybrid compound).\n'
            '3. DISEASE REQUIREMENTS: Explicitly list the specific target diseases this newly discovered formula would be required to treat, explaining why the modification makes it more effective.'
        ),
        expected_output=(
            'A comprehensive Scientific Discovery Report divided into three clear sections: '
            '### 1. Compound Suitability Evaluation, '
            '### 2. New Discovered Formula & Structure, and '
            '### 3. Target Disease Requirements & Therapeutic Mechanism.'
        ),
        agent=formulation_scientist
    )

def create_stability_task():
    return Task(
        description='Analyze the newly discovered formula provided by the Formulation Scientist. Stress-test this theoretical compound for: 1. THERMODYNAMIC STABILITY (shelf-life, environmental vulnerabilities). 2. PHYSIOLOGICAL SURVIVABILITY (first-pass metabolism, stomach acid pH ~1.5). 3. DELIVERY REQUIREMENTS (specific delivery mechanisms needed to keep the molecule intact).',
        expected_output='A Pharmacokinetic & Stability Report containing: ### 1. Degradation Vulnerabilities, ### 2. Physiological Survivability Assessment, and ### 3. Required Delivery Mechanisms.',
        agent=stability_tester
    )