import urllib.request
import urllib.parse
import json
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# 1. Define the strict schema for validation
class WikipediaSearchInput(BaseModel):
    """Input schema for the Wikipedia search tool."""
    query: str = Field(..., description="The medicine formula or drug name to search on Wikipedia.")

# 2. Define the custom Wikipedia Tool
class WikipediaSearchTool(BaseTool):
    name: str = "Wikipedia Search Tool"
    description: str = "Searches Wikipedia directly for authoritative data on chemical formulas, drugs, and pharmacology."
    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str) -> str:
        try:
            # Step 1: Query the Wikipedia Search API to find the most relevant article title
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json"
            
            req = urllib.request.Request(search_url, headers={'User-Agent': 'CrewAIMedicalAgent/1.0'})
            with urllib.request.urlopen(req) as response:
                search_data = json.loads(response.read().decode())
            
            results = search_data.get("query", {}).get("search", [])
            if not results:
                return f"No Wikipedia pages found matching the formula or term: '{query}'."
            
            # Pick the top search result
            best_title = results[0]["title"]
            
            # Step 2: Query the Wikipedia Content API to fetch the introduction summary text
            encoded_title = urllib.parse.quote(best_title)
            content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={encoded_title}&format=json"
            
            req_content = urllib.request.Request(content_url, headers={'User-Agent': 'CrewAIMedicalAgent/1.0'})
            with urllib.request.urlopen(req_content) as response:
                content_data = json.loads(response.read().decode())
            
            pages = content_data.get("query", {}).get("pages", {})
            for page_id, page_content in pages.items():
                if "extract" in page_content and page_content["extract"].strip():
                    return f"Source Article: Wikipedia - {best_title}\n\n{page_content['extract']}"
            
            return f"Found a Wikipedia page titled '{best_title}', but could not extract a clean summary introduction."
            
        except Exception as e:
            return f"An error occurred while connecting directly to Wikipedia: {str(e)}"