"""
04_str_output_parser.py
------------------------
Concept: Using StrOutputParser for plain text extraction
LangChain Class: StrOutputParser

This script demonstrates:
- How to get raw text output from an LLM.
- How to use StrOutputParser to extract and clean string content.
- A common use case: generating short summaries or responses in plain text.

Example Use Case:
Summarize a movie plot in one sentence.
"""

import os
import json
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# Setup
# ==========================================
load_dotenv()

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

# Initialize LLM dynamically
if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 200),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_output_tokens=model_cfg.get("max_output_tokens", 200),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Step 1: Define a simple text prompt
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["movie_name"],
    template="Summarize the plot of '{movie_name}' in one concise sentence."
)

# ==========================================
# Step 2: Build parsing pipeline
# ==========================================
parser = StrOutputParser()  # Extracts raw text

# ==========================================
# Step 3: Generate raw and parsed outputs
# ==========================================
movie = "Inception"
formatted_prompt = prompt_template.format(movie_name=movie)

print("🎬 String Output Parsing Demonstration")
print("--------------------------------------")
print(f"🔹 Input Movie: {movie}\n")

# Step 3a: Raw model output
raw_response = llm.invoke(formatted_prompt)
print("🔹 Raw Model Response Object:\n")
print(raw_response)
print("\n--------------------------------------\n")

# Step 3b: Parsed output (clean text)
parsed_response = parser.invoke(raw_response)
print("🔹 Parsed Text Output:\n")
print(parsed_response)
