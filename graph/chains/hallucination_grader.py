from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
import json
from typing import Dict, Any

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def _hallucination_grader_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom function to grade whether a generation is grounded in the provided documents.
    """
    # Make sure we're working with document objects properly
    documents = inputs.get("documents", [])
    generation = inputs.get("generation", "")
    
    # Extract document content if documents are objects
    doc_content = ""
    if documents:
        if hasattr(documents[0], 'page_content'):
            # If documents are Document objects
            doc_content = "\n\n".join([doc.page_content for doc in documents])
        elif isinstance(documents, str):
            # If documents are already a string
            doc_content = documents
        else:
            # Fallback
            doc_content = str(documents)
    
    # Create the prompt
    prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'true' or 'false'. 'true' means that the answer is grounded in / supported by the set of facts.

IMPORTANT: Your response MUST be a valid JSON object with a single field 'binary_score' that is a boolean (true or false).
For example: {{"binary_score": true}} or {{"binary_score": false}}

Do not include any other text or explanation in your response, only the JSON object.

Set of facts: 

{documents}

LLM generation: 

{generation}
"""
    
    # Format the prompt with the inputs
    formatted_prompt = prompt.format(documents=doc_content, generation=generation)
    
    # Get the response from the LLM
    response = llm.invoke(formatted_prompt)
    
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
            
            # Try cleaning up the JSON string - replace yes/no with true/false
            json_str = json_str.replace('"yes"', 'true').replace('"no"', 'false')
            json_str = json_str.replace("'yes'", 'true').replace("'no'", 'false')
            json_str = json_str.replace(': yes', ': true').replace(': no', ': false')
            
            try:
                result = json.loads(json_str)
                # Convert string values to boolean
                if isinstance(result.get("binary_score"), str):
                    if result["binary_score"].lower() in ["yes", "true"]:
                        result["binary_score"] = True
                    else:
                        result["binary_score"] = False
                return result
            except json.JSONDecodeError:
                pass
        
        # Last resort pattern matching
        if "binary_score" in content and any(x in content.lower() for x in ["yes", "true"]):
            return {"binary_score": True}
        elif "binary_score" in content and any(x in content.lower() for x in ["no", "false"]):
            return {"binary_score": False}
        
        # Fallback
        print("WARNING: Could not parse JSON from hallucination grader response. Using default value.")
        return {"binary_score": True}

# Create a Runnable version of the function
hallucination_grader = RunnableLambda(_hallucination_grader_fn)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
Your response should be a JSON object with a single field 'binary_score' that is a boolean."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
