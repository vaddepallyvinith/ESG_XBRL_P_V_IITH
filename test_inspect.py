import os
from parser import XBRLParser

file_path = "/home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/financial_dataset/TCS/XBRL/FY2023-24.xml"
parser = XBRLParser(file_path)
contexts, units, facts, namespaces = parser.parse()

# Search for any tag containing NameOf
print("Searching for NameOf concepts:")
for fact in facts:
    if "nameof" in fact.concept.lower():
        print(f"NameOf tag: {fact.concept} = {fact.value}")

print("\nSearching for any concept that has 'company' or 'entity' and value length < 100:")
for fact in facts:
    c_lower = fact.concept.lower()
    if ("company" in c_lower or "entity" in c_lower or "listed" in c_lower) and len(fact.value) < 100:
        if any(w in fact.value.lower() for w in ["tata", "consultancy", "reliance", "limited", "ltd"]):
            print(f"Matched tag: {fact.concept} = {fact.value}")
