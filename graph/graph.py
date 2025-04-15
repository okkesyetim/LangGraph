from dotenv import load_dotenv
from typing import Dict, Any
import os
import json
import time
import requests
import base64

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score["binary_score"]:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score["binary_score"]:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            # To prevent infinite loops, check if we've already tried web search
            if state.get("web_search_attempts", 0) >= 2:
                print("---MAX WEB SEARCH ATTEMPTS REACHED, RETURNING CURRENT GENERATION---")
                return "useful"  # Force end the loop
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    score = question_router.invoke({"question": question})
    datasource = score["datasource"]
    
    if datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        print("---WEB SEARCH ATTEMPT 1---")
        return WEBSEARCH
    else:
        print("---ROUTE QUESTION TO VECTORSTORE---")
        return RETRIEVE


def web_search_with_counter(state: GraphState) -> Dict[str, Any]:
    """Wrapper around web_search that tracks the number of attempts."""
    # Get the current number of attempts, defaulting to 0
    attempts = state.get("web_search_attempts", 0) + 1
    print(f"---WEB SEARCH ATTEMPT {attempts}---")
    
    # Call the actual web_search function
    result = web_search(state)
    
    # Update the attempts counter in the state
    result["web_search_attempts"] = attempts
    
    return result


# Create StateGraph without config parameter
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search_with_counter)  # Use the wrapper function

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# Compile the graph
app = workflow.compile()

# Create a manual Mermaid diagram based on our graph structure
mermaid_str = """
graph TD
    START([Start]) --> ROUTER{Route Question}
    ROUTER -->|Vectorstore| RETRIEVE[Retrieve Documents]
    ROUTER -->|Web Search| WEBSEARCH[Web Search]
    
    RETRIEVE --> GRADE_DOCUMENTS[Grade Documents]
    GRADE_DOCUMENTS -->|Documents Relevant| GENERATE[Generate Answer]
    GRADE_DOCUMENTS -->|Documents Not Relevant| WEBSEARCH
    
    WEBSEARCH --> GENERATE
    
    GENERATE -->|Check Hallucinations| HALLUCINATION{Hallucination Check}
    HALLUCINATION -->|Grounded| QUESTION_CHECK{Question Check}
    HALLUCINATION -->|Not Grounded| GENERATE
    
    QUESTION_CHECK -->|Addresses Question| END([End])
    QUESTION_CHECK -->|Doesn't Address| WEBSEARCH
"""

# Save the Mermaid diagram to a file
try:
    with open("graph.mmd", "w") as f:
        f.write(mermaid_str)
    print("Mermaid diagram saved to graph.mmd")
    
    # Provide instructions for visualization
    print("\nTo visualize the graph:")
    print("1. Visit https://mermaid.live/")
    print("2. Copy and paste the contents of graph.mmd")
    print("3. Download the PNG from the website")
    
    # Try to automatically open the Mermaid Live Editor with the diagram
    try:
        import webbrowser
        import urllib.parse
        
        encoded_mermaid = urllib.parse.quote(mermaid_str)
        url = f"https://mermaid.live/edit#pako:{encoded_mermaid}"
        
        print("\nAttempting to open Mermaid Live Editor in your browser...")
        webbrowser.open(url)
        print("If the browser didn't open automatically, please visit the URL manually.")
        
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        
except Exception as e:
    print(f"Error saving Mermaid diagram: {e}")