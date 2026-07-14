import os
import requests
from pathlib import Path

env_path = Path("/home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/.env")
with open(env_path, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip().strip("'\"")

api_key = os.environ.get("GEMINI_API_KEY", "")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)
print("Status:", response.status_code)
try:
    models = response.json().get('models', [])
    print("Found", len(models), "models.")
    for m in models:
        if 'gemini' in m['name'] and 'flash' in m['name']:
            print(m['name'], m.get('supportedGenerationMethods', []))
except Exception as e:
    print(response.text)
