import os

from llmfusion.base.exceptions import RateLimitError
from llmfusion.utils.key_manager import KeyManager
from llmfusion.providers import OpenAIClient
from llmfusion.base.models import LLMInput, LLMConfig
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize key manager with multiple API keys
key_manager = KeyManager(
    provider="openai",
    keys=["key1", "key2", "key3"]
)

# Create client with key manager
config = LLMConfig(
    model_name="gpt-4o",
    temperature=0.7,
    key_manager=key_manager  # Pass key manager to config
)
client = OpenAIClient(config)


# Example usage with monitoring
def process_requests(prompts: list[str]):
    for i, prompt in enumerate(prompts):
        try:
            # Get next available key
            key, cost = key_manager.get_next_key("gpt-4")
            print(f"Using key: {key.fingerprint()} (Cost: ${cost:.6f}/token)")

            # Make API call
            input = LLMInput(prompt=prompt)
            response = client.generate(input)

            # Record successful usage
            key_manager.record_usage(key, tokens=len(response.split()), success=True)
            print(f"Response {i + 1}: {response[:50]}...")

        except RateLimitError:
            print("Rate limit hit, rotating keys...")
            key_manager.disable_key(key.fingerprint())
            time.sleep(1)  # Backoff before retry
            continue

        except Exception as e:
            print(f"Error: {str(e)}")
            key_manager.record_usage(key, tokens=0, success=False)
            continue


# Generate some test prompts
prompts = [f"Explain concept {i} in simple terms" for i in range(1, 10)]

# Process requests
process_requests(prompts)

# Get usage report
report = key_manager.get_usage_report()
print("\nUsage Report:")
print(f"Total requests: {report['total_usage']['requests']}")
print(f"Total tokens: {report['total_usage']['tokens']}")
print(f"Total cost: ${report['total_usage']['cost']:.2f}")
print("\nKey Details:")
for fingerprint, stats in report['key_details'].items():
    print(f"Key {fingerprint}:")
    print(f"  Requests: {stats['usage_count']}")
    print(f"  Failures: {stats['failed_attempts']}")
    print(f"  Status: {'Active' if stats['is_active'] else 'Disabled'}")
    print(f"  Last Used: {time.ctime(stats['last_used'])}")
