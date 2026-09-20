"""
Prompt Loader Utility.

Loads prompt templates from markdown files in the prompts/ directory.
"""
from pathlib import Path
from typing import Optional

# Locate prompts directory relative to this file
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt template by name and format with provided kwargs.
    
    Args:
        name: Name of the prompt file (without .md extension)
        **kwargs: Variables to substitute in the template using str.format()
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> load_prompt("system_prompt", runtime_paths_context="...")
        >>> load_prompt("downstream_agent_system")
    """
    prompt_file = PROMPTS_DIR / f"{name}.md"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()
    
    # If kwargs provided, format the template
    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # If a placeholder is missing, return template with partial substitution
            # This allows for prompts that don't need all placeholders
            for key, value in kwargs.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
    
    return template


def get_available_prompts() -> list:
    """List all available prompt names."""
    if not PROMPTS_DIR.exists():
        return []
    return [f.stem for f in PROMPTS_DIR.glob("*.md")]
