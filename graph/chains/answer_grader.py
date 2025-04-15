from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import json

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def _answer_grader_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom function to grade whether an answer addresses a question.
    
    Args:
        inputs: Dictionary containing 'question' and 'generation'
        
    Returns:
        Dictionary with 'binary_score' key (boolean)
    """
    question = inputs.get("question", "")
    generation = inputs.get("generation", "")
    
    # Create the prompt
    prompt = f"""You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score true or false. true means that the answer resolves the question.

IMPORTANT: Your response MUST be a valid JSON object with a single field 'binary_score' that is a boolean (true or false).
For example: {{"binary_score": true}} or {{"binary_score": false}}

Do not include any other text or explanation in your response, only the JSON object.

User question: {question}

LLM Generation: {generation}
"""
    
    # Get the response from the LLM
    response = llm.invoke(prompt)
    
    # Extract the JSON from the response
    try:
        # Try to parse the response as JSON
        result = json.loads(response.content)
        return result
    except json.JSONDecodeError:
        # If parsing fails, try to extract JSON from the text
        content = response.content
        # Look for JSON-like patterns
        if "{" in content and "}" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: return a default response
        print("WARNING: Could not parse JSON from LLM response. Using default value.")
        return {"binary_score": True}

# Create a Runnable version of the function
answer_grader = RunnableLambda(_answer_grader_fn)
