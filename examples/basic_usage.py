import asyncio
import os

from dotenv import load_dotenv

from llmfusion.core import  LLMProvider
load_dotenv()
from llmfusion.base.models import LLMInput, LLMConfig
from llmfusion.providers import OpenAIClient, get_provider, ClaudeClient, GeminiClient, DeepSeekClient

config = LLMConfig(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",
    cache_ttl=3600,
    rate_limit_rpm=1000
)
#
# client = OpenAIClient(config)
#
# input = LLMInput(
#     prompt="Explain quantum computing",
#     system_prompt="You are a physics expert",
#     temperature=0.7,
#     max_tokens=500
# )
#

config = LLMConfig(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model_name="claude-3-5-sonnet-latest"
)
# client = ClaudeClient(config)
#
input = LLMInput(
    prompt="EYou are a physics professor. Explain quantum computing",
#    system_prompt="You are a physics professor",
    temperature=0.7,
    max_tokens=1000
)
#
# response = client.generate(input)
#
# print(response)


config = LLMConfig(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model_name="gemini-1.5-pro",
    safety_settings={
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
    }
)
# client = GeminiClient(config)
#
#
# response = client.generate(input)
#
# print(response)
#
#
# config = LLMConfig(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     model_name="deepseek-chat"
# )
# client = DeepSeekClient(config)
#
# # Synchronous call
# response = client.generate(LLMInput(
#     prompt="Write a Python function to calculate factorial",
#     temperature=0.3
# ))
#
#
# print(response)
# Streaming
# for chunk in client.stream(LLMInput(prompt="Explain recursion")):
#     print(chunk, end="")


# Initialize the provider client once
openai_provider = LLMProvider(
    provider="openai",
    model_name="gpt-4o",
    env_file=".env",  # Optional: Path to .env file
    cost_map_file="cost_map.json"  # Optional: Path to cost map JSON
)

# Make multiple API calls
response1 = openai_provider.get_completion("What is AI?")
print("Response 1:", response1)

response2 = openai_provider.get_completion("Explain quantum computing.")
print("Response 2:", response2)


# Get usage report
print("Usage Report:", openai_provider.get_usage_report())