import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, Process

# Import your cleanly separated, modular components
from agents import medical_researcher, formulation_scientist, compound_llm
from tasks import create_research_task, create_innovation_task

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ==========================================
# 2. UI PAGE SETUP
# ==========================================
st.set_page_config(page_title="AI Compound Explorer", page_icon="", layout="wide")

st.title(" AI Drug Discovery & Compound Explorer")
st.markdown("An agentic framework built on top of **Groq Compound Routing Architecture** to research medical elements and safely propose advanced structural optimizations.")

# ==========================================
# 3. SIDEBAR SYSTEM DASHBOARD
# ==========================================
with st.sidebar:
    st.header("Engine Infrastructure")
    st.markdown("**Target Model:** `ChatGroq(model_name='groq/compound')`")
    
    # Status Indicators
    if os.environ.get("GROQ_API_KEY"):
        st.success(" Groq Connection: Active")
    else:
        st.error(" Groq Connection: Disconnected")
        
    if os.environ.get("LANGCHAIN_API_KEY"):
        st.success(" LangSmith Pipeline: Linked")
    else:
        st.warning(" LangSmith Pipeline: Disabled")

    st.markdown("---")
    st.markdown("**Plan Guardrails:**\n-  Max Calls: `28 RPM` \n-  Agent Depth: `3 Steps Max` \n-  Document Cutoff: `1500 Chars` active")

# ==========================================
# 4. MAIN INTERACTIVE WORKSPACE
# ==========================================
formula_input = st.text_input(
    "Target Formula / Chemical Compound Input", 
    placeholder="e.g., C8H9NO2, Metformin, Amoxicillin, Ibuprofen..."
)

# Execute button
if st.button("Execute Groq Compound Pipeline", type="primary"):
    
    # Pre-flight checks
    if not os.environ.get("GROQ_API_KEY"):
        st.error(" Error: GROQ_API_KEY is missing from your .env file.")
        st.stop()
        
    if not formula_input.strip():
        st.warning(" Please provide a valid chemical compound formulation string or name.")
        st.stop()

    # Spinner locks the UI while the agents do the heavy lifting
    with st.spinner("Initializing Groq Compound routing and deploying agents..."):
        try:
            # 1. Generate tasks dynamically based on user input
            research_task = create_research_task(formula_input)
            innovation_task = create_innovation_task()

            # 2. Build Crew using your working ChatGroq instantiation
            drug_discovery_crew = Crew(
                agents=[medical_researcher, formulation_scientist],
                tasks=[research_task, innovation_task],
                process=Process.sequential,
                function_calling_llm=compound_llm, # Keeps everything uniform
                max_rpm=28, 
                verbose=False 
            )

            # 3. Fire pipeline execution
            result = drug_discovery_crew.kickoff()
            
            # 4. Display Results
            st.success("Pipeline Execution Complete!")
            st.markdown("###  Generated Output Proposal Data")
            st.info(result)

        except Exception as e:
            st.error(f" An error occurred during execution: {str(e)}")