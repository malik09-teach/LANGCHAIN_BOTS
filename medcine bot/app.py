import os
import streamlit as st
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from crewai import Crew, Process

# Import your cleanly separated, modular components
from agents import medical_researcher, formulation_scientist, stability_tester, compound_llm
from tasks import create_research_task, create_innovation_task, create_stability_task

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
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "current_formula" not in st.session_state:
    st.session_state.current_formula = ""

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
            stability_task = create_stability_task()

            # 2. Build Crew using your working ChatGroq instantiation
            drug_discovery_crew = Crew(
                agents=[medical_researcher, formulation_scientist, stability_tester],
                tasks=[research_task, innovation_task, stability_task],
                process=Process.sequential,
                function_calling_llm=compound_llm, # Keeps everything uniform
                max_rpm=28, 
                verbose=False 
            )

            # 3. Fire pipeline execution
            result = drug_discovery_crew.kickoff()
            
            # Save to session state
            st.session_state.pipeline_result = result
            st.session_state.current_formula = formula_input

        except Exception as e:
            st.error(f" An error occurred during execution: {str(e)}")

# ==========================================
# 5. RESULTS DASHBOARD
# ==========================================
if st.session_state.pipeline_result is not None:
    st.success("Pipeline Execution Complete!")
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Phase 1: Baseline Research", 
        "🧬 Phase 2: Structural Innovation", 
        "🛡️ Phase 3: Stability & Delivery"
    ])
    
    with tab1:
        st.markdown("This phase focuses on understanding the existing properties of the compound, its mechanisms, and known uses.")
    with tab2:
        st.markdown("This phase explores structural modifications to optimize the compound for enhanced efficacy against target diseases.")
    with tab3:
        st.markdown("This phase rigorously tests the modified compound for thermodynamic stability, physiological survivability, and delivery needs.")
        
    st.markdown("---")
    st.markdown("### Generated Output Proposal Data")
    
    st.info(st.session_state.pipeline_result)
    
    st.download_button(
        label="Download Complete Report",
        data=str(st.session_state.pipeline_result),
        file_name=f"{st.session_state.current_formula}_analysis.txt" if st.session_state.current_formula else "pipeline_report.txt",
        mime="text/plain"
    )