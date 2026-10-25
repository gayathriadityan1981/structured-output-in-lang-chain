"""
06_structured_output_parser.py
------------------------------
Concept: Enforcing schema-based structured responses
LangChain Classes: ResponseSchema, StructuredOutputParser

This script demonstrates:
- How to define field-level schemas for LLM output.
- How to build a StructuredOutputParser to enforce consistent response formatting.
- How to show format instructions (for prompt injection into the model).
- How to parse and print structured JSON output.

Example Use Case:
Extract key details (product name, category, price, sentiment) from a customer review.
"""

import os
import json
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema
from langchain_classic.output_parsers import StructuredOutputParser
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

# Initialize LLM dynamically
if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.3),
        max_tokens=model_cfg.get("max_tokens", 400),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.3),
        max_output_tokens=model_cfg.get("max_output_tokens", 400),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Step 1: Define Schema for structured output
# ==========================================
schemas = [
    ResponseSchema(name="product_name", description="Name of the product being reviewed."),
    ResponseSchema(name="category", description="Product category or type."),
    ResponseSchema(name="price", description="Approximate price mentioned in the review, if any."),
    ResponseSchema(name="sentiment", description="Overall sentiment of the review. One of: positive, negative, neutral."),
    ResponseSchema(name="summary", description="One-line summary of the review.")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

print("-----------------------------------------------------------")
print(format_instructions)
print("-----------------------------------------------------------")

# ==========================================
# Step 2: Define prompt
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["review"],
    template=(
        "You are an assistant that extracts structured information from product reviews.\n\n"
        "Customer Review:\n{review}\n\n"
        "Please return your answer in the following format:\n{format_instructions}"
    )
)

# Create final formatted prompt
customer_review = (
    "I recently bought the Sony WH-1000XM5 headphones, and they are absolutely amazing! "
    "The sound quality is superb, noise cancellation is unbeatable, though they’re quite expensive at around $400."
)

formatted_prompt = prompt_template.format(
    review=customer_review, format_instructions=format_instructions
)

# ==========================================
# Step 3: Run model
# ==========================================
print("🧾 Structured Output Parser Demonstration")
print("----------------------------------------")
print("🔹 Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n----------------------------------------\n")

raw_response = llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n----------------------------------------\n")

# ==========================================
# Step 4: Parse structured output
# ==========================================
try:
    parsed_output = parser.invoke(raw_response.content)
    print("🔹 Parsed Structured Output:\n")
    print(json.dumps(parsed_output, indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")