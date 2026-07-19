from crewai import Agent
from tools import extract_text_from_url

def create_agents():
    # Agent 1: Web Content Extractor
    content_extractor = Agent(
        role='Web Content Extractor',
        goal='Extract raw text content from the given URL accurately.',
        backstory='You are an expert at fetching and extracting clean text from web pages using custom tools.',
        verbose=True,
        allow_delegation=False,
        tools=[extract_text_from_url]
    )

    # Agent 2: Content Summarizer
    content_summarizer = Agent(
        role='Content Summarizer',
        goal='Summarize the extracted web content into a concise, easily readable format.',
        backstory='You are a skilled editor who excels at digesting long texts and creating insightful, comprehensive summaries.',
        verbose=True,
        allow_delegation=False
    )

    return content_extractor, content_summarizer
