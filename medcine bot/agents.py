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
    role='Lead Pharmacological Researcher',
    goal='Search Wikipedia directly for the given medicine formula, understand its current uses, mechanism of action, and summarize the findings.',
    backstory='You are a brilliant pharmacologist at a top-tier research institute. You extract clear medical profiles from raw chemical configurations.',
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