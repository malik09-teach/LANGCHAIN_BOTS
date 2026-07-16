import os
from crewai import Agent
from langchain_groq import ChatGroq
from tools import WikipediaSearchTool

# Instantiate the Groq Compound model using LangChain's native class
compound_llm = ChatGroq(
    temperature=0.2, 
    model_name="groq/llama-3.3-70b-versatile" 
)

# Instantiate your custom tool
wiki_tool = WikipediaSearchTool()

medical_researcher = Agent(
    role='Medical Diagnostician & Pharmacological Researcher',
    goal='Analyze the provided symptoms, identify potential diseases, and recommend standard medicinal compounds used to treat them.',
    backstory='You are a brilliant medical diagnostician and pharmacologist at a top-tier research institute. You extract clear diagnoses from symptoms and propose baseline pharmacological treatments.',
    llm=compound_llm, # <--- Passing the working ChatGroq object here
    verbose=True,
    allow_delegation=False,
    respect_context_window=True,
    max_iter=3, 
    tools=[wiki_tool]
)

formulation_scientist = Agent(
    role='Innovative Formulation Scientist',
    goal='Propose a new, theoretical medicine composition or structural modification based on the research provided to better cure target diseases.',
    backstory='You are a visionary biochemist. You take structural data and safely look for variants or molecular enhancements to optimize efficacy.',
    llm=compound_llm, # <--- Passing the working ChatGroq object here
    verbose=True,
    respect_context_window=True,
    max_iter=3, 
    allow_delegation=False
)

stability_tester = Agent(
    role='Chemical Stability & Pharmacokinetics Analyst',
    goal='Critically evaluate the thermodynamic stability, shelf-life, and physiological survivability of the proposed compound.',
    backstory='You are a rigorous physical chemist and pharmacokinetic expert. You analyze molecular bonds to predict how a drug will survive in different environments, including the human digestive tract, blood plasma, and long-term shelf storage. You are highly critical of fragile molecular structures.',
    llm=compound_llm,
    verbose=True,
    allow_delegation=False,
    respect_context_window=True,
    max_iter=3
)