import re
import json
import time
import httpx
from pathlib import Path
from typing import Dict, Any, Optional
import models
from config import (
    ENVIRONMENTAL_KEYWORDS, SOCIAL_KEYWORDS, GOVERNANCE_KEYWORDS,
    GEMINI_API_KEY, GEMINI_MODEL, OUTPUT_DIR
)
from utils import logger

CACHE_PATH = OUTPUT_DIR / "concept_category_cache.json"

class BRSRDataTransformer:
    def __init__(self):
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, str]:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading category cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving category cache: {e}")

    def normalize_and_categorize(self, fact: models.Fact) -> models.Fact:
        """Normalize the value and categorize the concept of the fact."""
        # 1. Normalize value
        val_clean = fact.value.strip()
        
        # Check missing values
        if not val_clean or val_clean in ["-", "Nil", "NIL", "NA", "Not Applicable", "Not applicable", "null", "None"]:
            fact.normalized_value = None
            fact.value_type = "text"
        # Check booleans
        elif val_clean.lower() in ["yes", "true", "y"]:
            fact.normalized_value = True
            fact.value_type = "boolean"
        elif val_clean.lower() in ["no", "false", "n"]:
            fact.normalized_value = False
            fact.value_type = "boolean"
        # Check dates (YYYY-MM-DD or DD-MM-YYYY)
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', val_clean):
            fact.normalized_value = val_clean
            fact.value_type = "date"
        elif re.match(r'^\d{2}-\d{2}-\d{4}$', val_clean):
            # Convert DD-MM-YYYY to YYYY-MM-DD
            parts = val_clean.split("-")
            fact.normalized_value = f"{parts[2]}-{parts[1]}-{parts[0]}"
            fact.value_type = "date"
        # Check numbers (including scientific notation and commas)
        else:
            # Strip commas and check if numeric
            num_str = val_clean.replace(",", "")
            # Check standard float/int or scientific notation
            if re.match(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', num_str):
                try:
                    if "." in num_str or "e" in num_str.lower():
                        fact.normalized_value = float(num_str)
                    else:
                        fact.normalized_value = int(num_str)
                    
                    # Deduce if percentage
                    if "percentage" in fact.concept.lower() or "rate" in fact.concept.lower() or fact.unit_ref == "pure":
                        fact.value_type = "percentage"
                    else:
                        fact.value_type = "numeric"
                except ValueError:
                    fact.normalized_value = val_clean
                    fact.value_type = "text"
            else:
                # Fallback to text
                fact.normalized_value = val_clean
                fact.value_type = "text"

        # 2. Categorize concept
        fact.category = self._categorize_concept(fact.concept)
        
        return fact

    def _categorize_concept(self, concept: str) -> str:
        """Categorize concept using keywords -> cache -> Gemini API."""
        c_lower = concept.lower()
        
        # Step 1: Rule-based keyword matching (Fast)
        # Environmental
        if any(kw in c_lower for kw in ENVIRONMENTAL_KEYWORDS):
            return "Environmental"
        # Social
        if any(kw in c_lower for kw in SOCIAL_KEYWORDS):
            return "Social"
        # Governance
        if any(kw in c_lower for kw in GOVERNANCE_KEYWORDS):
            return "Governance"

        # Step 2: Check Cache
        if concept in self.cache:
            return self.cache[concept]

        # Step 3: LLM Fallback (Gemini API)
        category = self._query_gemini_category(concept)
        
        # Save to cache
        self.cache[concept] = category
        self._save_cache()
        
        return category

    def _query_gemini_category(self, concept: str) -> str:
        """Call Gemini API using requests to classify a concept."""
        if not GEMINI_API_KEY:
            logger.debug(f"Gemini API Key not set. Categorizing '{concept}' as Other.")
            return "Other"
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        prompt = (
            f"You are an ESG taxonomy expert. Classify the following BRSR (Business Responsibility and Sustainability Report) "
            f"XBRL concept tag name into one of the four categories: \"Environmental\", \"Social\", \"Governance\", or \"Other\".\n\n"
            f"Concept tag name: {concept}\n\n"
            f"Respond with exactly one word from: [\"Environmental\", \"Social\", \"Governance\", \"Other\"]. "
            f"Do not include any other text, prefix, or explanation."
        )
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 10
            }
        }
        
        max_retries = 3
        backoff_factor = 2.0
        current_delay = 1.5

        for attempt in range(max_retries):
            try:
                logger.info(f"Querying Gemini (Attempt {attempt+1}/{max_retries}) to classify ambiguous concept: '{concept}'")
                response = httpx.post(url, headers=headers, json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # Clean up any potential markdown or punctuation
                    clean_text = re.sub(r'[^a-zA-Z]', '', text)
                    valid_categories = ["Environmental", "Social", "Governance", "Other"]
                    for cat in valid_categories:
                        if clean_text.lower() == cat.lower():
                            return cat
                    logger.warning(f"Gemini returned unexpected category string: '{text}'. Falling back to Other.")
                    break
                    
                elif response.status_code in [429, 503]:
                    logger.warning(f"Gemini API returned temporary error {response.status_code}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(f"Gemini API returned error: {response.status_code} - {response.text}")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to query Gemini API on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    
        return "Other"
