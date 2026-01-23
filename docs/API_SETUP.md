# API Setup Guide

## Overview

MCH experiments require API access to three vendors. You can run with any subset of vendors.

## Cost Estimation

| Experiment | Trials | Estimated Cost |
|------------|--------|----------------|
| Minimal Replication | 10/model | ~$2.50 |
| Medical Domain | 50/model | ~$10 |
| Full Philosophy | 100/model | ~$17 |

Average cost per trial: ~$0.028

## OpenAI Setup

1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Add to `config/api_keys.yaml`:
   ```yaml
   openai_api_key: "sk-..."
   ```

**Rate Limits:**
- GPT-4o-mini: 500 RPM (requests per minute)
- GPT-4o: 500 RPM

## Google (Gemini) Setup

1. Go to https://aistudio.google.com/app/apikey
2. Create API key
3. Add to `config/api_keys.yaml`:
   ```yaml
   google_api_key: "AI..."
   ```

## Anthropic (Claude) Setup

1. Go to https://console.anthropic.com/settings/keys
2. Create new API key
3. Add to `config/api_keys.yaml`:
   ```yaml
   anthropic_api_key: "sk-ant-..."
   ```

**Rate Limits:**
- Claude Haiku: 50 RPM
- Claude Opus: 50 RPM

## Environment Variables (Alternative)

Instead of `api_keys.yaml`, you can use environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AI..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file in the project root.

## Testing Your Setup

```bash
python -c "
import openai
import anthropic
import google.generativeai as genai
print('All imports successful!')
"
```

## Troubleshooting

**"Rate limit exceeded"**
- The scripts include automatic retry with exponential backoff
- Default: 5-second delay between calls

**"API key invalid"**
- Check for extra spaces in your key
- Ensure the key has the correct permissions

**"Safety filter blocked"**
- Some content may trigger vendor safety filters
- Retry with rephrased prompt if needed
