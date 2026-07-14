import json
from pathlib import Path

processed_dir = Path("/home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/data/processed")
files_to_check = [
    "GRI_GRI 2_ General Disclosures 2021.json",
    "GRI_GRI 302_ Energy 2016.json",
    "GRI_GRI 11_ Oil and Gas Sector 2021 V1.1.json"
]

for fname in files_to_check:
    fpath = processed_dir / fname
    if not fpath.exists(): 
        print(f"File not found: {fname}")
        continue
    with open(fpath) as f: 
        data = json.load(f)
        
    print(f"\n{'='*50}")
    print(f"File: {fname}")
    print(f"{'='*50}")
    
    stds = data.get("standards", [])
    valid_discs = 0
    
    for s in stds:
        discs = s.get("disclosures", [])
        for d in discs:
            reqs = d.get("requirements", [])
            if len(reqs) > 0:
                print(f"  {d.get('id')} -> {len(reqs)} req blocks")
                valid_discs += 1
                
    print(f"\nTotal valid disclosures (with reqs): {valid_discs}")
