import os
from crewai import Crew, Process
from agents import get_data_engineer, get_model_architect
from tasks import get_analysis_task, get_modeling_task

# Set up your language model API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

def run_ml_agent_pipeline(target_csv):
    # Instantiate the agents
    data_eng = get_data_engineer()
    model_arch = get_model_architect()
    
    # Instantiate tasks sequentially
    task1 = get_analysis_task(data_eng, target_csv)
    task2 = get_modeling_task(model_arch)
    
    # Assemble into a sequential Crew
    ml_crew = Crew(
        agents=[data_eng, model_arch],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )
    
    print(f"Executing Sequential Crew for: {target_csv}...")
    result = ml_crew.kickoff()
    return result

if __name__ == "__main__":
    # Example placeholder dataset path
    dataset_file = "sample_data.csv"
    
    # Run the pipeline
    final_code_output = run_ml_agent_pipeline(dataset_file)
    
    print("\n================== GENERATED CODE OUTPUT ==================\n")
    print(final_code_output)