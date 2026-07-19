from crewai import Task

def create_tasks(extractor_agent, summarizer_agent, target_url):
    # Task 1: Extract content
    extract_task = Task(
        description=f'Extract the text content from the following URL using the Webpage Text Extractor tool: {target_url}',
        expected_output='The raw extracted text from the webpage.',
        agent=extractor_agent
    )

    # Task 2: Summarize content
    summarize_task = Task(
        description='Read the raw extracted text provided by the previous task and summarize it into a concise, well-structured 3-paragraph summary.',
        expected_output='A clear, well-structured 3-paragraph summary of the extracted content.',
        agent=summarizer_agent
    )

    return [extract_task, summarize_task]
