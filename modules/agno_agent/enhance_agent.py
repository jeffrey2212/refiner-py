from textwrap import dedent
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.ollama import Ollama
from ollama import Client as OllamaClient
from modules.utils.qdrant import get_similar_prompts

class EnhancedPrompt(BaseModel):
    prompt: str = Field(
        ...,
        description="The enhanced prompt with quality boosters and style elements"
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of the enhancements made"
    )
    reference_prompts: List[str] = Field(
        default=[],
        description="Similar prompts that were used as reference"
    )

client = OllamaClient(host='http://jeffaiserver:11434')

instruction = dedent("""\
    You are a Stable Diffusion prompt engineering expert. Your task is to help users create and refine prompts for Stable Diffusion image generation.

    Key points:
    1. "1girl" is a common tag meaning "one female character"
    2. Prompts should be comma-separated tags and descriptions
    3. Quality boosters like "masterpiece, best quality" are common
    4. Negative prompts help avoid unwanted elements
    5. Each model may have specific style preferences

    When enhancing prompts:
    1. Analyze common elements in successful prompts
    2. Identify key style elements
    3. Consider useful parameters from similar prompts
    4. Incorporate relevant elements while maintaining user's intent
    5. Keep character and scene descriptions clear and detailed

    Your response should be valid JSON in this format:
    {
        "prompt": "enhanced, comma-separated, prompt",
        "explanation": "Brief explanation of changes",
        "reference_prompts": ["similar prompt 1", "similar prompt 2"]
    }
    """)

agent = Agent(
    model=Ollama(id="llama3.2:latest", client=client),
    instructions=instruction,
    response_model=EnhancedPrompt,
    add_history_to_messages=True,
    num_history_responses=3,
    structured_outputs=True
)

def enhance_prompt(keywords: List[str], model_name: str = "general") -> Tuple[str, Dict[str, Any]]:
    """
    Enhance a prompt using similar prompts from Qdrant and LLM refinement
    
    Args:
        keywords: List of selected keywords to enhance
        model_name: Name of the model to get style-specific prompts
        
    Returns:
        Tuple of (enhanced prompt, debug info)
    """
    try:
        # Create base prompt from keywords
        base_prompt = ", ".join(keywords)
        
        # Get similar prompts from Qdrant using Agno's integration
        similar_prompts = get_similar_prompts(base_prompt, model_name, k=3)
        
        # Create prompt for the LLM
        context = "\n".join([f"Similar prompt: {p}" for p in similar_prompts])
        prompt = f"""Base prompt: {base_prompt}

        Here are some similar high-quality prompts for reference:
        {context}

        Please enhance this prompt while maintaining its core elements.
        Add quality boosters, style elements, and clear descriptions.
        Explain your enhancements."""
        
        # Get enhanced prompt from LLM
        response = agent.run(prompt)
        if response and response.content:
            debug_info = {
                "base_prompt": base_prompt,
                "similar_prompts": similar_prompts,
                "llm_prompt": prompt,
                "raw_response": response.content.dict(),
                "explanation": response.content.explanation
            }
            return response.content.prompt, debug_info
            
    except Exception as e:
        print(f"Error enhancing prompt: {str(e)}")
        return base_prompt, {"error": str(e)}
    
    return base_prompt, {}