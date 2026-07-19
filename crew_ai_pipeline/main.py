import os
import sys
from crewai import Crew, Process
from agents import create_agents
from tasks import create_tasks

def main():
    # Example URL to test if none provided as argument
    target_url = "https://example.com"
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    
    print(f"Targeting URL: {target_url}")

    # Note: Ensure you have your GROQ_API_KEY set in your environment variables, 
    # or configure your LLM as needed for CrewAI.
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable is not set.")
        print("Please set it to run the CrewAI agents successfully.")

    # 1. Create Agents
    extractor, summarizer = create_agents()

    # 2. Create Tasks
    tasks = create_tasks(extractor, summarizer, target_url)

    # 3. Form the Crew
    crew = Crew(
        agents=[extractor, summarizer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    # 4. Execute Pipeline
    print("\nStarting CrewAI pipeline...\n")
    try:
        result = crew.kickoff()
        
        print("\n######################")
        print("FINAL SUMMARY RESULT:")
        print("######################\n")
        print(result)
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")

if __name__ == "__main__":
    main()
