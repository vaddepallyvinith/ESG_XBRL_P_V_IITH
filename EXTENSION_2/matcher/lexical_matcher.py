import logging
import re
from typing import List, Dict, Tuple
from matcher.models import OntologyConcept

logger = logging.getLogger(__name__)

import logging
import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from matcher.models import OntologyConcept

logger = logging.getLogger(__name__)

class LexicalMatcher:
    """AML-inspired Lexical Matcher.
    Computes Jaccard similarity over synonym-expanded word tokens and SequenceMatcher ratio over aliases.
    """
    def __init__(self, config: dict):
        self.config = config.get("lexical_matcher", {})
        self.weight_label = self.config.get("weight_label", 0.95)
        self.weight_exact = self.config.get("weight_exact", 1.0)
        
        # ESG Synonym mappings
        self.SYNONYM_MAP = {
            "ghg": "greenhouse gas",
            "co2": "carbon dioxide",
            "emissions": "emission",
            "discharges": "discharge",
            "releases": "release",
            "effluents": "wastewater",
            "effluent": "wastewater",
            "sewage": "wastewater",
            "workers": "employee",
            "worker": "employee",
            "employees": "employee",
            "labor": "employee",
            "workforce": "employee",
            "staff": "employee",
            "anticorruption": "anti-corruption",
            "remuneration": "compensation",
            "salary": "compensation",
            "wages": "compensation",
            "pay": "compensation",
            "electricity": "energy",
            "power": "energy",
            "fuel": "energy",
        }

    def _normalize(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        # Multi-word synonym normalization
        text = re.sub(r'\bgreenhouse\s+gas(?:es)?\b', 'ghg', text)
        text = re.sub(r'\bcarbon\s+dioxide\b', 'co2', text)
        text = re.sub(r'\banti\s*\-\s*corruption\b', 'anticorruption', text)
        text = re.sub(r'\banti\s+corruption\b', 'anticorruption', text)
        text = re.sub(r'\bhuman\s+rights\b', 'humanrights', text)
        text = re.sub(r'\bwater\s+consumption\b', 'waterconsumption', text)
        text = re.sub(r'\bwater\s+withdrawal\b', 'waterwithdrawal', text)
        text = re.sub(r'\bwater\s+withdrawal\s+and\s+consumption\b', 'wateruse', text)
        text = re.sub(r'\boccupational\s+health\s+and\s+safety\b', 'ohs', text)
        
        words = re.findall(r'\w+', text)
        stopwords = {"and", "or", "the", "a", "of", "in", "to", "for", "with", "on", "at", "by", "from", "an", "is", "are"}
        normalized = []
        for w in words:
            if w not in stopwords:
                w_canon = self.SYNONYM_MAP.get(w, w)
                normalized.append(w_canon)
        return normalized

    def get_aliases(self, label: str) -> List[str]:
        aliases = [label]
        lbl_lower = label.lower()
        if "ghg" in lbl_lower:
            aliases.append(lbl_lower.replace("ghg", "greenhouse gas"))
            aliases.append(lbl_lower.replace("ghg", "greenhouse gases"))
        if "greenhouse gas" in lbl_lower:
            aliases.append(lbl_lower.replace("greenhouse gas", "ghg"))
        if "greenhouse gases" in lbl_lower:
            aliases.append(lbl_lower.replace("greenhouse gases", "ghg"))
        if "anti-corruption" in lbl_lower:
            aliases.append(lbl_lower.replace("anti-corruption", "anticorruption"))
            aliases.append(lbl_lower.replace("anti-corruption", "anti corruption"))
        if "anticorruption" in lbl_lower:
            aliases.append(lbl_lower.replace("anticorruption", "anti-corruption"))
        if "anti corruption" in lbl_lower:
            aliases.append(lbl_lower.replace("anti corruption", "anti-corruption"))
        if "occupational health and safety" in lbl_lower:
            aliases.append(lbl_lower.replace("occupational health and safety", "ohs"))
            aliases.append(lbl_lower.replace("occupational health and safety", "health and safety"))
        if "ohs" in lbl_lower:
            aliases.append(lbl_lower.replace("ohs", "occupational health and safety"))
        return list(set(aliases))

    def match(self, source_concepts: List[OntologyConcept], target_concepts: List[OntologyConcept]) -> Dict[Tuple[str, str], float]:
        logger.info("Running Lexical Matcher...")
        scores = {}
        
        # Speed optimization: pre-calculate tokens and aliases
        source_aliases = {c.uri: self.get_aliases(c.label) for c in source_concepts}
        target_aliases = {c.uri: self.get_aliases(c.label) for c in target_concepts}
        
        source_tokens = {
            c.uri: [self._normalize(alias) for alias in source_aliases[c.uri]]
            for c in source_concepts
        }
        target_tokens = {
            c.uri: [self._normalize(alias) for alias in target_aliases[c.uri]]
            for c in target_concepts
        }

        for s_concept in source_concepts:
            s_aliases = source_aliases[s_concept.uri]
            s_tok_lists = source_tokens[s_concept.uri]
            
            for t_concept in target_concepts:
                t_aliases = target_aliases[t_concept.uri]
                t_tok_lists = target_tokens[t_concept.uri]
                
                # Check for exact matches among any alias pair
                exact_match = False
                for sa in s_aliases:
                    for ta in t_aliases:
                        if sa.strip().lower() == ta.strip().lower():
                            exact_match = True
                            break
                    if exact_match:
                        break
                        
                if exact_match:
                    scores[(s_concept.uri, t_concept.uri)] = float(self.weight_exact)
                    continue
                
                # Otherwise, find the best similarity score across all alias pairs
                best_jaccard = 0.0
                best_seq_match = 0.0
                
                for s_toks in s_tok_lists:
                    s_set = set(s_toks)
                    if not s_set:
                        continue
                    for t_toks in t_tok_lists:
                        t_set = set(t_toks)
                        if not t_set:
                            continue
                        
                        intersection = s_set.intersection(t_set)
                        union = s_set.union(t_set)
                        jaccard = len(intersection) / len(union) if union else 0.0
                        if jaccard > best_jaccard:
                            best_jaccard = jaccard
                            
                for sa in s_aliases:
                    for ta in t_aliases:
                        seq_sim = SequenceMatcher(None, sa, ta).ratio()
                        if seq_sim > best_seq_match:
                            best_seq_match = seq_sim
                            
                lex_score = 0.5 * best_jaccard + 0.5 * best_seq_match
                score = lex_score * self.weight_label
                
                if score > 0.0:
                    scores[(s_concept.uri, t_concept.uri)] = float(score)
                    
        return scores
