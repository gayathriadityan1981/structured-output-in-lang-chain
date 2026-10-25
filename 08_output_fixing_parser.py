"""
08_output_fixing_parser.py
--------------------------
Concept: Automatically repairing malformed outputs.
LangChain Class: OutputFixingParser

This script demonstrates:
- How to wrap a base parser (e.g., PydanticOutputParser) with an OutputFixingParser.
- How LangChain auto-fixes invalid JSON or schema mismatches.
- How to compare the raw vs fixed outputs.

Example Use Case:
A model returns slightly malformed JSON — OutputFixingParser repairs it automatically.
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_classic.output_parsers import PydanticOutputParser,OutputFixingParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

if provider == "openai":
    base_llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.5),
        max_tokens=model_cfg.get("max_tokens", 400),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    base_llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.5),
        max_output_tokens=model_cfg.get("max_output_tokens", 400),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Step 1: Define Pydantic schema
# ==========================================
class MovieDetails(BaseModel):
    title: str = Field(..., description="Movie title.")
    year: int = Field(..., description="Release year.")
    genre: str = Field(..., description="Genre of the movie.")
    rating: float = Field(..., description="IMDB rating (0-10).")


# Base Pydantic parser
pydantic_parser = PydanticOutputParser(pydantic_object=MovieDetails)

# OutputFixingParser wraps the base parser
fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=base_llm)

# For LLM prompt hinting
format_instructions = pydantic_parser.get_format_instructions()


# ==========================================
# Step 2: Create intentionally “messy” output scenario
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["movie"],
    template=(
        "Extract structured details about the movie '{movie}'.\n\n"
        "IMPORTANT: You MUST intentionally create a JSON syntax error!\n"
        "For example:\n"
        "- Missing a comma between fields\n"
        "- Unclosed quote around a string\n"
        "- Extra trailing comma\n"
        "- Missing a required field entirely\n\n"
        "DO NOT output valid JSON. Make a deliberate mistake.\n\n"
        "Here's the schema you should have followed (but remember, MAKE AN ERROR):\n"
        "{format_instructions}"
    )
)

movie = "The Matrix (1999)"
formatted_prompt = prompt_template.format(movie=movie, format_instructions=format_instructions)


# ==========================================
# Step 3: Get raw model output and auto-fix it
# ==========================================
print("🧩 Output Fixing Parser Demonstration")
print("-------------------------------------")
print("🔹 Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n-------------------------------------\n")

# Get raw response (may be slightly invalid)
raw_response = base_llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n-------------------------------------\n")

# Use fixing parser — it re-prompts LLM internally to repair invalid JSON
try:
    fixed_output = fixing_parser.invoke(raw_response.content)
    print("🔹 Fixed & Parsed Output (Pydantic validated):\n")
    print(json.dumps(fixed_output.model_dump(), indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")
