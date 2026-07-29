import os
import sys
import argparse

# Force Python to look in the folder where main.py actually lives
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core elements
from crewai import Crew, Process, LLM
from agents import get_data_engineer, get_model_architect
from tasks import get_analysis_task, get_modeling_task

# Initialize the Groq LLM object correctly
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

def run_ml_agent_pipeline(target_csv):
    # 1. Instantiate the agents and explicitly give them the Groq engine
    data_eng = get_data_engineer()
    model_arch = get_model_architect()
    
    data_eng.llm = groq_llm
    model_arch.llm = groq_llm
    
    # 2. Instantiate tasks sequentially
    task1 = get_analysis_task(data_eng, target_csv)
    task2 = get_modeling_task(model_arch)
    
    # 3. Assemble into a strictly SEQUENTIAL Crew
    ml_crew = Crew(
        agents=[data_eng, model_arch],
        tasks=[task1, task2],
        process=Process.sequential,  # Clean linear chain execution
        verbose=True
    )
    
    print(f"\nExecuting Sequential Crew for: {target_csv}...\n")
    result = ml_crew.kickoff()
    return result

if __name__ == "__main__":
    # Command line argument parser entry point
    parser = argparse.ArgumentParser(description="Run the ML Setup Agent Pipeline.")
    parser.add_argument(
        "csv_path", 
        type=str, 
        nargs="?", 
        default="sample_data.csv",
        help="Path to the target CSV file (defaults to sample_data.csv)"
    )
    args = parser.parse_args()
    
    dataset_file = args.csv_path

    # Verify key is present
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    # Verify file exists
    if not os.path.exists(dataset_file):
        print(f"ERROR: The file '{dataset_file}' was not found.")
        sys.exit(1)
    
    # Execute
    final_code_output = run_ml_agent_pipeline(dataset_file)
    
    print("\n================== GENERATED CODE OUTPUT ==================\n")
    print(final_code_output)