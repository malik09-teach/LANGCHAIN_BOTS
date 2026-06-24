from crewai import Task
from agents import medical_researcher, formulation_scientist

def create_research_task():
    return Task(
        description='Search the web for the medical formula/drug: {medicine_formula}. Identify its standard name, its primary uses, side effects, and the diseases it currently cures. Provide a comprehensive summary.',
        expected_output='A detailed summary including the drug name, chemical formula, mechanism of action, treated diseases, and common side effects.',
        agent=medical_researcher
    )

def create_innovation_task():
    return Task(
        description='Analyze the pharmacological summary provided by the Researcher. Based on its mechanism of action and limitations, propose a NEW, theoretical chemical composition, modification, or drug combination. Explain how this new formulation works and what specific diseases it could theoretically cure more effectively.',
        expected_output='A scientific proposal for a new drug formulation, detailing the new composition, its theoretical mechanism of action, and the specific diseases it targets.',
        agent=formulation_scientist
    )