Here's a comprehensive `README.md` file for your LLM Providers library:


# LLM Providers

[![PyPI Version](https://img.shields.io/pypi/v/llm-providers)](https://pypi.org/project/llm-providers/)
[![Python Version](https://img.shields.io/pypi/pyversions/llm-providers)](https://pypi.org/project/llm-providers/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/yourusername/llm-providers/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/llm-providers/actions)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://yourusername.github.io/llm-providers/)

A unified Python library for interacting with multiple Large Language Model (LLM) providers with a consistent interface.

## Features

- **Multi-Provider Support**: OpenAI, Claude, Gemini, Groq, DeepSeek, HuggingFace (API and local)
- **Unified Interface**: Consistent API across all providers
- **Async Support**: Full async/await implementation
- **Streaming**: Real-time text generation
- **Caching**: Disk-based caching with TTL
- **Rate Limiting**: Built-in rate limiting
- **Error Handling**: Comprehensive error handling and retries
- **Type Safety**: Pydantic models for input validation
- **Extensible**: Easy to add new providers

## Installation

```bash
pip install llm-providers
```

## Supported Providers

| Provider     | API Support | Local Models | Streaming | Async |
|--------------|-------------|--------------|-----------|-------|
| OpenAI       | ✅          | ❌           | ✅        | ✅    |
| Claude       | ✅          | ❌           | ❌        | ✅    |
| Gemini       | ✅          | ❌           | ❌        | ✅    |
| Groq         | ✅          | ❌           | ✅        | ✅    |
| DeepSeek     | ✅          | ❌           | ✅        | ✅    |
| HuggingFace  | ❌          | ❌           | ❌        | ❌    |

## Quick Start

### Basic Usage

```python
from llmfusion import get_provider, LLMConfig, LLMInput

# Configure OpenAI
openai_config = LLMConfig(
    api_key="your-openai-key",
    model_name="gpt-4"
)

# Create client
openai = get_provider("openai", openai_config)

# Create input
input = LLMInput(
    prompt="Explain quantum computing",
    system_prompt="You are a physics expert",
    temperature=0.7,
    max_tokens=500
)

# Generate response
response = openai.generate(input)
print(response)
```

### Async Usage

```python
import asyncio
from llmfusion import get_provider, LLMConfig, LLMInput


async def main():
    config = LLMConfig(
        api_key="your-api-key",
        model_name="claude-3"
    )
    claude = get_provider("claude", config)

    input = LLMInput(prompt="Write a poem about AI")
    response = await claude.agenerate(input)
    print(response)


asyncio.run(main())
```

### Streaming

```python
from llmfusion import get_provider, LLMConfig, LLMInput

config = LLMConfig(
    api_key="your-api-key",
    model_name="gpt-4"
)
gpt = get_provider("openai", config)

input = LLMInput(prompt="Explain blockchain technology")
for chunk in gpt.stream(input):
    print(chunk, end="", flush=True)
```

## Configuration

### LLMConfig Parameters

| Parameter         | Type    | Default       | Description                          |
|-------------------|---------|---------------|--------------------------------------|
| `api_key`         | str     | None          | Provider API key                     |
| `model_name`      | str     | Required      | Model name/identifier                |
| `base_url`        | str     | Provider default | Custom API endpoint                |
| `timeout`         | int     | 30            | Request timeout in seconds           |
| `max_retries`     | int     | 3             | Maximum retry attempts               |
| `cache_ttl`       | int     | 3600          | Cache time-to-live in seconds        |
| `rate_limit_rpm`  | int     | 1000          | Requests per minute limit            |
| `device`          | str     | "auto"        | Device for local models (cpu/cuda)   |

## Advanced Usage

### Custom Cache Directory

```python
from llmfusion import set_cache_dir

set_cache_dir("/path/to/cache")
```

### Disable Caching

```python
config = LLMConfig(
    api_key="your-key",
    model_name="gpt-4",
    cache_ttl=0  # Disable caching
)
```

### Using Local HuggingFace Models

```python
config = LLMConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1"
)
hf = get_provider("huggingface", config)
```

## Error Handling

The library provides comprehensive error handling with custom exceptions:

```python
from llmfusion import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderError
)

try:
    response = client.generate(input)
except RateLimitError as e:
    print("Rate limit exceeded:", str(e))
except AuthenticationError as e:
    print("Authentication failed:", str(e))
except ProviderError as e:
    print("Provider error:", str(e))
except LLMError as e:
    print("General error:", str(e))
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Documentation

Full documentation is available at [https://github.com/mbenhaddou/llm_providers](https://github.com/mbenhaddou/llm_providers)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by the need for a unified LLM interface
- Built with ❤️ by Mohamed Ben Haddou