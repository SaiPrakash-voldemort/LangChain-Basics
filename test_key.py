from dotenv import load_dotenv
import os
import requests
import json

# Load .env
load_dotenv()

# Read key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ API key not found.")
    exit()

print(f"✅ API Key loaded: {api_key[:10]}...")

# OpenRouter endpoint
url = "https://openrouter.ai/api/v1/chat/completions"

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",  # must be included
    "X-Title": "Test Script"
}

# Payload
payload = {
    "model": "openai/gpt-3.5-turbo",  # use known working model
    "messages": [
        {"role": "user", "content": "Say hello"}
    ]
}

# Make request
try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        print("✅ Success!")
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(f"❌ Error {response.status_code}:")
        print(response.text)

except requests.exceptions.Timeout:
    print("❌ Request timed out. Try again later.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
