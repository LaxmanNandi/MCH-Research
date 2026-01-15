from anthropic import Anthropic
import os
from dotenv import load_dotenv
import sys

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Test different model names including Claude 4
models_to_test = [
    "claude-opus-4-20250514",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307"
]

for model in models_to_test:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'test'"}]
        )
        print(f"OK {model} works!")
        print(f"  Response: {message.content[0].text[:50]}")
        break  # Stop on first working model
    except Exception as e:
        print(f"FAIL {model}: {str(e)[:100]}")
