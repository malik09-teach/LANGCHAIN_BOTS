import os
import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from tools import WikipediaSearchTool

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
# Load variables from the local .env file
load_dotenv()

# Explicitly map keys to environment variables for CrewAI and LangChain
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", )
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-drug-discovery-gradio")

# ==========================================
# 2. CORE EXECUTION FUNCTION
# ==========================================
def run_crew_pipeline(medicine_formula):
    # Validation checks
    if not os.environ.get("GROQ_API_KEY"):
        return "❌ Error: GROQ_API_KEY is missing from your .env file."
    if not medicine_formula.strip():
        return "⚠️ Please enter a valid chemical formula or medicine name."

    try:
        # Initialize LLM with the latest versatile model
        groq_llm = ChatGroq(
            temperature=0.2, 
            model_name="groq/llama3-70b-8192" 
        )

        # Initialize the validated custom Wikipedia tool instance
        wiki_tool_instance = WikipediaSearchTool()

        # Define Agents
        medical_researcher = Agent(
            role='Lead Pharmacological Researcher',
            goal='Search Wikipedia directly for the given medicine formula, understand its current uses, mechanism of action, and summarize the findings.',
            backstory='You are a brilliant pharmacologist at a top-tier research institute. You excel at looking up raw chemical compositions on Wikipedia and extracting concrete medical profiles.',
            llm=groq_llm,
            verbose=True,
            allow_delegation=False,
            tools=[wiki_tool_instance]
        )

        formulation_scientist = Agent(
            role='Innovative Formulation Scientist',
            goal='Propose a new, theoretical medicine composition or structural modification based on the research provided to better cure target diseases.',
            backstory='You are a visionary biochemist. You take existing Wikipedia breakthroughs and safely brainstorm structural variants or combinations to optimize effectiveness.',
            llm=groq_llm,
            verbose=True,
            allow_delegation=False
        )

        # Define Tasks
        research_task = Task(
            description=f'Search Wikipedia for the medical formula/drug: {medicine_formula}. Identify its standard name, its primary uses, side effects, and the diseases it currently cures. Provide a comprehensive summary.',
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
            function_calling_llm=groq_llm,
            verbose=False 
        )

        # Execute pipeline
        result = drug_discovery_crew.kickoff()
        return str(result)

    except Exception as e:
        return f"❌ An error occurred during execution: {str(e)}"

# ==========================================
# 3. GRADIO UI DESIGN
# ==========================================
# Check status of API keys for the system indicator
groq_status = "✅ Active" if os.environ.get("GROQ_API_KEY") else "❌ Missing"
smith_status = "✅ Connected" if os.environ.get("LANGCHAIN_API_KEY") else "⚠️ Inactive (Traces Disabled)"

with gr.Blocks(title="AI Drug Discovery Pipeline") as demo:
    gr.Markdown("# 🧪 AI Drug Discovery & Formulation Pipeline")
    gr.Markdown("Submit a chemical formula or common drug name. The Lead Researcher will poll Wikipedia, parse the content, and hand it off to the Formulation Scientist to generate a theoretical modification.")
    
    with gr.Row():
        # System status card
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ System Status")
            gr.Markdown(f"**Groq Engine LLM:** {groq_status}")
            gr.Markdown(f"**LangSmith Telemetry:** {smith_status}")
            gr.Markdown("---")
            gr.Markdown("**Pipeline Architecture:**\n1. `WikipediaSearchTool` ➡️ \n2. `Lead Researcher Agent` ➡️ \n3. `Formulation Scientist Agent` ➡️ \n4. LangSmith Trace Log Generated.")
        
        # Main interactive workspace
        with gr.Column(scale=2):
            formula_input = gr.Textbox(
                label="Medicine Formula / Drug Name", 
                placeholder="e.g., C8H9NO2, Metformin, Ibuprofen...",
                lines=1
            )
            
            submit_btn = gr.Button("Run Agentic Pipeline", variant="primary")
            
            output_display = gr.Markdown(
                label="Final Formulation Proposal",
                value="*Results will appear here after the agent workflow runs completely.*"
            )

    # Set up button action with a native loading spinner animation
    submit_btn.click(
        fn=run_crew_pipeline,
        inputs=formula_input,
        outputs=output_display,
        show_progress="full"
    )

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Launch local server
    demo.launch(server_name="127.0.0.1", server_port=7860)











'''import streamlit as st
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
            st.error(f"An error occurred during execution: {e}")'''