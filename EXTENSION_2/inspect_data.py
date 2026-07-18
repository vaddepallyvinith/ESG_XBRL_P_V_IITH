import json
import pandas as pd

repo_path = "data/processed/mapping/mapping_repository.json"
with open(repo_path, 'r') as f:
    mappings = json.load(f)

print(f"Number of mappings in repository: {len(mappings)}")
if mappings:
    print("Example mapping keys:", mappings[0].keys())
    print("Example llm_verification:", set(m.get('llm_verification') for m in mappings))

results_path = "data/processed/mapping/multi_llm_results.json"
with open(results_path, 'r') as f:
    multi_res = json.load(f)

for model, data in multi_res.items():
    decisions = data.get("decisions", [])
    print(f"Model {model}: {len(decisions)} decisions")
    print(f"  First 10 decisions: {decisions[:10]}")
    print(f"  Unique decisions: {set(decisions)}")
