from smartfunc import backend
from typing import Final, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL: Final[str] = os.getenv("MODEL") or "llama3.2"

client: Final[OpenAI] = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

llm_backend: Final[object] = backend(
    client,
    model=MODEL,
    system="""
    You are an AI agent for a weather API. Provide short, friendly, and to-the-point responses. 
    Avoid unnecessary words. Do not use Markdown or any formatting—plain text only. 
    Always keep replies as brief and clear as possible. 
    Follow this style strictly.
    """,
)

@llm_backend
def analyze(api: Dict, prompt: str):
    return f"Analyze the data {api} and do {prompt}"

@llm_backend
def bye(text: str):
    return (
        "Check if the text is a bye message such as \"bye\" or \"goodbye\" or "
        "\"see you later\" or \"good night\" or \"good evening\" etc. "
        "If it is, return 1, otherwise return 0. "
        "JUST RETURN THE NUMBER, NOT ANYTHING ELSE.\n"
        f"Text: {text}"
    )


    
def isBye(text: str) -> bool:
    response: str = bye(text)
    output: bool = bool(int(response)) or False
    return output