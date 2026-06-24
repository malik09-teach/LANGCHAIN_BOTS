import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
# Load variables from the .env file
load_dotenv()

# Explicitly set the OS environment variables so LangChain and CrewAI can see them
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# LangSmith Observability setup
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# ==========================================
# 2. UI CONFIGURATION & SIDEBAR
# ==========================================
st.set_page_config(page_title="AI Drug Discovery Pipeline", page_icon="", layout="wide")

st.title(" AI Drug Discovery & Formulation Pipeline")
st.markdown("Enter a chemical formula to trigger the Agentic AI workflow. The Researcher will scour the web, and the Innovator will propose a new formulation.")
st.caption("Live monitoring is enabled via LangSmith.")

with st.sidebar:
    st.header(" System Status")
    
    # Simple status check to verify keys are loaded
    if os.environ["GROQ_API_KEY"]:
        st.success("Groq API Key Loaded")
    else:
        st.error(" Groq API Key Missing in .env")
        
    if os.environ["LANGCHAIN_API_KEY"]:
        st.success("LangSmith Tracking Active")
    else:
        st.warning("⚠️ LangSmith Key Missing (Tracking Disabled)")

    st.markdown("---")
    st.markdown("**Agents in this Crew:**")
    st.markdown("-  **Lead Researcher** (Web Search Enabled)")
    st.markdown("-  **Formulation Scientist** (Logic & Innovation)")

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================
formula_input = st.text_input("Enter Medicine Formula (e.g., C8H9NO2 for Paracetamol):", placeholder="C8H9NO2")

if st.button("Run AI Discovery Pipeline", type="primary"):
    
    if not os.environ["GROQ_API_KEY"]:
        st.error("Please add your GROQ_API_KEY to the .env file to proceed.")
        st.stop()
        
    if not formula_input:
        st.warning("Please enter a valid chemical formula.")
        st.stop()

    with st.spinner("Initializing Agents and searching pharmacological databases..."):
        try:
            # Initialize Tools and LLM
            search_tool = DuckDuckGoSearchRun()
            groq_llm = ChatGroq(
                temperature=0.2, 
                model_name="llama3-70b-8192"
            )

            # Define Agents
            medical_researcher = Agent(
                role='Lead Pharmacological Researcher',
                goal='Search the web and Wikipedia for the given medicine formula, understand its current uses, mechanism of action, and summarize the findings.',
                backstory='You are a brilliant pharmacologist working at a top-tier research institute. You excel at taking raw chemical names or formulas, scouring databases, and summarizing exactly what the drug does and what diseases it currently treats.',
                llm=groq_llm,
                verbose=True,
                allow_delegation=False,
                tools=[search_tool] 
            )

            formulation_scientist = Agent(
                role='Innovative Formulation Scientist',
                goal='Propose a new, theoretical medicine composition or structural modification based on the research provided to better cure target diseases.',
                backstory='You are a visionary biochemist. You look at existing medical formulas and find ways to improve them—either by combining them with other compounds, modifying their chemical structure to reduce side effects, or targeting new diseases entirely.',
                llm=groq_llm,
                verbose=True,
                allow_delegation=False
            )

            # Define Tasks
            research_task = Task(
                description=f'Search the web for the medical formula/drug: {formula_input}. Identify its standard name, its primary uses, side effects, and the diseases it currently cures. Provide a comprehensive summary.',
                expected_output='A detailed summary including the drug name, chemical formula, mechanism of action, treated diseases, and common side effects.',
                agent=medical_researcher
            )

            innovation_task = Task(
                description='Analyze the pharmacological summary provided by the Researcher. Based on its mechanism of action and limitations, propose a NEW, theoretical chemical composition, modification, or drug combination. Explain how this new formulation works and what specific diseases it could theoretically cure more effectively.',
                expected_output='A scientific proposal for a new drug formulation, detailing the new composition, its theoretical mechanism of action, and the specific diseases it targets.',
                agent=formulation_scientist
            )

            # Assemble Crew
            drug_discovery_crew = Crew(
                agents=[medical_researcher, formulation_scientist],
                tasks=[research_task, innovation_task],
                process=Process.sequential, 
                verbose=False 
            )

            # Execute
            result = drug_discovery_crew.kickoff()

            # Output
            st.success("Research and Formulation Complete!")
            st.markdown("### 🔬 Final Formulation Proposal")
            st.info(result)

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")