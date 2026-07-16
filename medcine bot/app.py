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
st.set_page_config(page_title="AI Medical Diagnostician & Drug Discovery", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(139, 92, 246, 0.4);
    }
    
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.3);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 6px;
        padding: 10px 20px;
        color: #94a3b8;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(139, 92, 246, 0.2) !important;
        border-bottom: 2px solid #8b5cf6 !important;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 AI Medical Diagnostician & Drug Discovery")
st.markdown("<p style='font-size: 1.2rem; color: #cbd5e1; margin-bottom: 2rem;'>An agentic framework built on top of <b>Groq</b> to analyze patient symptoms, diagnose potential diseases, recommend treatments, and safely propose advanced structural drug optimizations.</p>", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR SYSTEM DASHBOARD
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050431.png", width=100)
    st.markdown("## System Dashboard")
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Infrastructure")
    st.info("**Target Model:**\n`ChatGroq(model_name='groq/compound')`")
    
    # Status Indicators
    if os.environ.get("GROQ_API_KEY"):
        st.success("🟢 Groq Connection: Active")
    else:
        st.error("🔴 Groq Connection: Disconnected")
        
    if os.environ.get("LANGCHAIN_API_KEY"):
        st.success("🟢 LangSmith Pipeline: Linked")
    else:
        st.warning("🟠 LangSmith Pipeline: Disabled")

    st.markdown("---")
    st.markdown("### 🛡️ Plan Guardrails")
    st.markdown("- 🚦 Max Calls: `28 RPM` \n- 🧠 Agent Depth: `3 Steps Max` \n- 📄 Document Cutoff: `1500 Chars`")

# ==========================================
# 4. MAIN INTERACTIVE WORKSPACE
# ==========================================
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "current_symptoms" not in st.session_state:
    st.session_state.current_symptoms = ""

symptoms_input = st.text_input(
    "Target Symptoms Input", 
    placeholder="e.g., chronic cough, shortness of breath, fatigue..."
)

# Execute button
if st.button("Execute Groq Compound Pipeline", type="primary"):
    
    # Pre-flight checks
    if not os.environ.get("GROQ_API_KEY"):
        st.error(" Error: GROQ_API_KEY is missing from your .env file.")
        st.stop()
        
    if not symptoms_input.strip():
        st.warning(" Please provide a valid set of symptoms.")
        st.stop()

    # Spinner locks the UI while the agents do the heavy lifting
    with st.spinner("Initializing Groq Compound routing and deploying agents..."):
        try:
            # 1. Generate tasks dynamically based on user input
            research_task = create_research_task(symptoms_input)
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
            st.session_state.current_symptoms = symptoms_input

        except Exception as e:
            st.error(f" An error occurred during execution: {str(e)}")

# ==========================================
# 5. RESULTS DASHBOARD
# ==========================================
if st.session_state.pipeline_result is not None:
    st.success("Pipeline Execution Complete!")
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Phase 1: Diagnosis & Baseline Treatments", 
        "🧬 Phase 2: Structural Innovation", 
        "🛡️ Phase 3: Stability & Delivery"
    ])
    
    with tab1:
        st.info("💡 **Phase 1** focuses on analyzing the provided symptoms, diagnosing potential diseases, and identifying standard baseline treatments.")
    with tab2:
        st.info("🚀 **Phase 2** explores structural modifications to optimize the compound for enhanced efficacy against target diseases.")
    with tab3:
        st.info("⚖️ **Phase 3** rigorously tests the modified compound for thermodynamic stability, physiological survivability, and delivery needs.")
        
    st.markdown("---")
    st.markdown("### 🔬 Generated Output Proposal Data")
    
    st.info(st.session_state.pipeline_result)
    
    st.download_button(
        label="Download Complete Report",
        data=str(st.session_state.pipeline_result),
        file_name=f"{st.session_state.current_symptoms}_analysis.txt" if st.session_state.current_symptoms else "pipeline_report.txt",
        mime="text/plain"
    )