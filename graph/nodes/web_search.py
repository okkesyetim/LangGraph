from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.documents import Document
from graph.state import GraphState


def web_search(state: GraphState) -> Dict[str, Any]:
    """Search the web for relevant information."""
    print("---WEB SEARCH---")
    question = state["question"]
    
    # Make sure environment variables are loaded
    load_dotenv()
    
    # Get the API key from environment
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("WARNING: TAVILY_API_KEY not found in environment variables")
    else:
        # Set the environment variable for TavilySearchAPIWrapper
        os.environ["TAVILY_API_KEY"] = tavily_api_key
    
    # Initialize the search wrapper (it will use the environment variable)
    search = TavilySearchAPIWrapper()
    
    try:
        # Perform the search
        docs = search.results(question)
        
        # Handle different possible response structures
        web_results = ""
        if docs and isinstance(docs, list):
            if all(isinstance(d, dict) for d in docs):
                # Check if 'content' key exists in the dictionaries
                if all('content' in d for d in docs):
                    web_results = "\n".join([d["content"] for d in docs])
                # Check if 'text' key exists in the dictionaries
                elif all('text' in d for d in docs):
                    web_results = "\n".join([d["text"] for d in docs])
                else:
                    # Fallback: convert each dictionary to string
                    web_results = "\n".join([str(d) for d in docs])
            else:
                # If docs are not dictionaries, convert them to strings
                web_results = "\n".join([str(d) for d in docs])
        else:
            # If docs is not a list or is empty, convert it to string
            web_results = str(docs)
        
        # Create a Document object for consistency with other parts of the system
        doc = Document(page_content=web_results)
        
        # Replace existing documents with the new web search result
        print("--- Replacing documents with web search results ---")
        return {
            "documents": [doc],
            "question": question,
            "web_search": True
        }
    
    except Exception as e:
        print(f"Error during web search: {e}")
        # Return a fallback response if search fails
        fallback_content = f"Unable to perform web search due to error: {str(e)}. Providing fallback document."
        doc = Document(page_content=fallback_content)
        
        # Replace existing documents even on error, providing the fallback
        print("--- Replacing documents with fallback due to web search error ---")
        return {
            "documents": [doc],
            "question": question,
            "web_search": True
        }
