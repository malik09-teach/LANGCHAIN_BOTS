from langchain_community.tools import DuckDuckGoSearchRun

def get_search_tool():
    """Returns the DuckDuckGo search tool for web browsing."""
    return DuckDuckGoSearchRun()