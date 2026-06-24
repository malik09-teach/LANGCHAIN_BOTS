from crewai import Agent
from langchain_groq import ChatGroq
from tools import WikipediaSearchTool

# Initialize the Groq LLM (Ensure you are using an active model string)
groq_llm = ChatGroq(
    temperature=0.2, 
    model_name="llama-3.3-70b-versatile" 
)

# Instantiate the newly designed Wikipedia tool
wiki_tool_instance = WikipediaSearchTool()

# Define the Researcher equipped with Wikipedia
medical_researcher = Agent(
    role='Lead Pharmacological Researcher',
    goal='Search Wikipedia directly for the given medicine formula, understand its current uses, mechanism of action, and summarize the findings.',
    backstory='You are a brilliant pharmacologist working at a top-tier research institute. You excel at taking raw chemical names or formulas, searching Wikipedia directly, and summarizing exactly what the drug does and what diseases it currently treats.',
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
    tools=[wiki_tool_instance] # <--- Your clean Wikipedia instance goes here
)

# Define the Innovator
formulation_scientist = Agent(
    role='Innovative Formulation Scientist',
    goal='Propose a new, theoretical medicine composition or structural modification based on the research provided to better cure target diseases.',
    backstory='You are a visionary biochemist. You look at existing medical formulas and find ways to improve them—either by combining them with other compounds, modifying their chemical structure to reduce side effects, or targeting new diseases entirely.',
    llm=groq_llm,
    verbose=True,
    allow_delegation=False
)