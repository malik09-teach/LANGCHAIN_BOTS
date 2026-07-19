import os
from crewai import Agent
from langchain_groq import ChatGroq
from tools import extract_text_from_url

def create_agents():
    # Initialize the Groq LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192", 
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # Agent 1: Web Content Extractor
    content_extractor = Agent(
        role='Web Content Extractor',
        goal='Extract raw text content from the given URL accurately.',
        backstory='You are an expert at fetching and extracting clean text from web pages using custom tools.',
        verbose=True,
        allow_delegation=False,
        tools=[extract_text_from_url],
        llm=llm
    )

    # Agent 2: Content Summarizer
    content_summarizer = Agent(
        role='Content Summarizer',
        goal='Summarize the extracted web content into a concise, easily readable format.',
        backstory='You are a skilled editor who excels at digesting long texts and creating insightful, comprehensive summaries.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return content_extractor, content_summarizer
