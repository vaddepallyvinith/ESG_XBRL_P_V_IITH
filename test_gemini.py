import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:10]}...")

models_to_test = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

for model in models_to_test:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": "Say Hello"}]}]
    }
    try:
        r = httpx.post(url, headers=headers, json=payload)
        print(f"Model: {model} -> Status: {r.status_code}")
        if r.status_code == 200:
            print(f"Response: {r.json()['contents'][0]['parts'][0]['text'].strip()}")
            break
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Failed: {e}")
