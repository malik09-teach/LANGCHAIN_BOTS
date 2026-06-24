from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the underlying LangChain tool
search_engine = DuckDuckGoSearchRun()

@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """
    Searches the web for pharmacological data, chemical formulas, or general information.
    Always use this when you need to find up-to-date facts on the internet.
    """
    # Use .invoke() to run the search and return the text string
    return search_engine.invoke(query)