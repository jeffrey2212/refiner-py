from textwrap import dedent
from typing import List
import re
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from ollama import Client as OllamaClient

# Safe emoji mapping for common categories
SAFE_EMOJIS = {
    'human': ['👤', '👨', '👩', '👶', '👴', '👵'],
    'animal': ['🐕', '🐈', '🐦', '🐠', '🐢', '🐘'],
    'object': ['📦', '📱', '📚', '🎸', '🎨', '🎮'],
    'scenery': ['🌅', '🌊', '🌳', '🏔️', '🏖️', '🌆']
}

def clean_emoji_text(text: str) -> str:
    # Check if text has invalid unicode or problematic characters
    try:
        # Try to find emoji and word pattern
        match = re.match(r'^([\s\S]*?)\s+(\w+.*?)$', text.strip())
        if not match:
            return f"📍 {text.strip()}"  # Default emoji if no pattern found
            
        emoji_part, word_part = match.groups()
        
        # If emoji part has invalid characters, replace with a safe emoji
        if '<0x' in emoji_part or len(emoji_part.encode()) > 8:
            # Choose an appropriate safe emoji based on the word
            word_lower = word_part.lower()
            for category, emojis in SAFE_EMOJIS.items():
                if category in word_lower:
                    return f"{emojis[0]} {word_part}"
            return f"📍 {word_part}"  # Default if no category matches
            
        return text.strip()
    except:
        return f"📍 {text.strip()}"  # Fallback for any errors

class RefinerOptions(BaseModel):
    options: List[str] = Field(
        ..., 
        description="List of 6 options related to the given keywords. Each option should be a specific word with its most relevant emoji.",
        min_items=6,
        max_items=6
    )
    category: str = Field(
        ..., 
        description="The main category or theme these options belong to, derived from the keywords"
    )
    description: str = Field(
        ..., 
        description="A brief description of how these options relate to the previous keywords"
    )

client = OllamaClient(
    host='http://jeffaiserver:11434',
)

instruction = dedent("""\
        You are a helpful assistant, guiding the user in building text-to-image prompts.
        The process is iterative: you provide options, and the user chooses one to refine the prompt.

        User will pick one of the following options:
        1. 👤 Human
        2. 🐕 Animal
        3. 📦 Object/Item
        4. 🌅 Scenery/Environment

        After the user selects a category, you will provide 6 options related to that category.
        Each option should be presented with a simple emoji followed by a descriptive word.
        
        Examples of options with emojis:
        - For Human: 👨 Man, 👩 Woman, 👶 Baby, 👴 Elder
        - For Animal: 🐕 Dog, 🐈 Cat, 🐦 Bird, 🐠 Fish
        - For Object: 📱 Phone, 📚 Book, 🎸 Guitar, 🎨 Art
        - For Scenery: 🌅 Sunset, 🌊 Ocean, 🏔️ Mountain, 🌳 Forest

        If user asks for a reroll, provide 6 new options with emojis related to the last selected category.
        After each selection, briefly recap the user's choice so far and then provide the next set of options.

        If the user input is not valid, gently guide them back to the available options.
        """)

modelid= "llama3.2:latest"

agent = Agent(
    model=Ollama(id=modelid, client=client), 
    instructions=instruction,
    response_model=RefinerOptions,
    add_history_to_messages=True,
    num_history_responses=7,
    structured_outputs=True
)

def generate_options(selected_keywords: List[str]) -> tuple[List[str], dict]:
    if not selected_keywords:
        return [], {}
        
    prompt = f"""Based on these keywords: {', '.join(selected_keywords)}
    Generate 6 new options that would complement and refine these keywords.
    Each option MUST be in the format: [emoji] [word]
    Use simple emojis followed by descriptive words.
    Make each option specific and meaningful.
    
    IMPORTANT: Your response must be valid JSON in this exact format:
    {{
        "options": ["emoji word1", "emoji word2", ...],
        "category": "brief category name",
        "description": "brief description"
    }}"""
    
    try:
        response: RunResponse = agent.run(prompt)
        if not response or not response.content:
            return [], {"error": "Empty response from LLM"}
            
        if isinstance(response.content, str):
            # If we got a string instead of structured data, try to extract options
            import re
            options = []
            for line in response.content.split('\n'):
                # Look for emoji + word pattern
                match = re.search(r'([^\w\s][\u200d\u2600-\u27ff\ud83c-\ud83e][\ufe0e\ufe0f]?(?:[\u2600-\u27ff\ud83c-\ud83e][\ufe0e\ufe0f]?)*)[\s]*([^\n,]+)', line)
                if match:
                    emoji, word = match.groups()
                    options.append(f"{emoji.strip()} {word.strip()}")
            
            if options:
                cleaned_options = [clean_emoji_text(opt) for opt in options[:6]]
                debug_info = {
                    "raw_response": response.content,
                    "extracted_options": options,
                    "cleaned_options": cleaned_options,
                    "warning": "Response was not in expected JSON format"
                }
                return cleaned_options, debug_info
                
        # Normal case - structured response
        if hasattr(response.content, 'options'):
            cleaned_options = [clean_emoji_text(opt) for opt in response.content.options]
            debug_info = {
                "raw_options": response.content.options,
                "category": getattr(response.content, 'category', 'Unknown'),
                "description": getattr(response.content, 'description', 'No description'),
                "prompt": prompt,
                "cleaned_options": cleaned_options
            }
            return cleaned_options, debug_info
            
        return [], {"error": f"Unexpected response format: {type(response.content)}"}
            
    except Exception as e:
        import traceback
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "raw_response": getattr(response, 'content', None) if 'response' in locals() else None
        }
        print(f"Error generating options: {str(e)}")
        return [], error_info
    
    return [], {}
