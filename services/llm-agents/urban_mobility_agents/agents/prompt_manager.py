import os
from string import Template
from loguru import logger

class PromptManager:
    """Manager to load and format prompts from external markdown files."""
    
    def __init__(self, prompts_dir: str):
        self.prompts_dir = prompts_dir
        self._cache = {}

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        if prompt_name not in self._cache:
            filepath = os.path.join(self.prompts_dir, f"{prompt_name}.md")
            if not os.path.exists(filepath):
                logger.error(f"Prompt file not found: {filepath}")
                raise FileNotFoundError(f"Prompt file not found: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                self._cache[prompt_name] = Template(f.read())
        
        template = self._cache[prompt_name]
        return template.safe_substitute(**kwargs)