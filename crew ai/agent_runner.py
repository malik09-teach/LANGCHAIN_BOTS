import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

def log_step(step_output):
    # Ensure file is created or appended to properly
    with open('updates.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Agent Step:\n")
        f.write(str(step_output) + "\n\n")

def run_agent():
    # Clear previous updates
    with open('updates.txt', 'w', encoding='utf-8') as f:
        f.write("Agent started...\n\n")

    # Clear previous final post
    if os.path.exists('final_post.txt'):
        os.remove('final_post.txt')

    search_tool = DuckDuckGoSearchRun()
    
    researcher = Agent(
        role='Tech Researcher',
        goal='Find the latest developments in AI technology',
        backstory='You are a seasoned data researcher who uncovers hidden tech trends.',
        verbose=True,
        tools=[search_tool],
        step_callback=log_step
    )
    
    writer = Agent(
        role='Tech Blogger',
        goal='Craft engaging summaries of technical research',
        backstory='You are a renowned writer capable of explaining complex topics simply.',
        verbose=True,
        step_callback=log_step
    )
    
    research_task = Task(
        description='Conduct research on the newest AI agent frameworks.',
        expected_output='A bulleted list of the top frameworks and their key features.',
        agent=researcher
    )
    
    writing_task = Task(
        description='Use the research list to write a short, engaging blog post.',
        expected_output='A 2-paragraph blog post formatted in markdown.',
        agent=writer,
        output_file='final_post.txt'
    )
    
    tech_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential
    )
    
    # Run the crew
    result = tech_crew.kickoff()
    
    # Add a final log step
    with open('updates.txt', 'a', encoding='utf-8') as f:
        f.write("\nAgent Finished.\n")
        
    return result

if __name__ == "__main__":
    print(run_agent())
