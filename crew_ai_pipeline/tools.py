from langchain.tools import tool
import requests
from bs4 import BeautifulSoup

@tool("Webpage Text Extractor")
def extract_text_from_url(url: str) -> str:
    """
    Useful to extract and ingest raw text content from a specific web page URL.
    Input should be a fully qualified URL (e.g., https://example.com).
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text[:5000] 
    except Exception as e:
        return f"Error extracting data from {url}: {str(e)}"
