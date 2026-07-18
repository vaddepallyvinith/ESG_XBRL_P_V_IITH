import logging
from typing import List
from matcher.models import MappingCandidate

logger = logging.getLogger(__name__)

import logging
import re
from typing import List
from matcher.models import MappingCandidate

logger = logging.getLogger(__name__)

class OntologyReasoner:
    """Ontology Reasoner.
    Applies logical rules, checking consistency and resolving disjointness constraints (Mapping Repair).
    """
    def __init__(self, config: dict):
        self.config = config.get("reasoner", {})

    def classify_concept_esg(self, concept) -> str:
        # returns "E", "S", "G", or "Unknown"
        full_text = (concept.uri + " " + concept.label + " " + " ".join(concept.hierarchy_path)).lower()
        
        # 1. Check GRI standard number patterns (e.g., GRI 302, GRI 403, etc.)
        gri_match = re.search(r'gri\s*(\d{3})', full_text)
        if gri_match:
            std_num = int(gri_match.group(1))
            if 300 <= std_num < 400:
                return "E"
            elif 400 <= std_num < 500:
                return "S"
            elif 200 <= std_num < 300:
                return "G"
                
        # 2. Check BRSR Principle patterns (e.g., Principle 6, Principle 3, etc.)
        prin_match = re.search(r'principle\s*(\d)', full_text)
        if prin_match:
            p_num = int(prin_match.group(1))
            if p_num in [6]:
                return "E"
            elif p_num in [3, 4, 5, 8, 9]:
                return "S"
            elif p_num in [1, 7]:
                return "G"
            elif p_num == 2:
                return "E" # Sustainable products
                
        # 3. Fallback to keyword heuristics if we can't find standard IDs
        env_words = ["environment", "emission", "energy", "water", "waste", "climate", "pollution", "biodiversity", "greenhouse", "ghg", "carbon", "co2", "effluent", "spill"]
        soc_words = ["employee", "labor", "human right", "social", "gender", "safety", "health", "diversity", "community", "worker", "training", "salary", "wage", "accident", "injury"]
        gov_words = ["governance", "anti-corruption", "board", "ethics", "shareholder", "compliance", "tax", "bribery", "corruption", "whistleblower", "advocacy", "policy"]
        
        e_score = sum(w in full_text for w in env_words)
        s_score = sum(w in full_text for w in soc_words)
        g_score = sum(w in full_text for w in gov_words)
        
        if e_score > s_score and e_score > g_score:
            return "E"
        elif s_score > e_score and s_score > g_score:
            return "S"
        elif g_score > e_score and g_score > s_score:
            return "G"
            
        return "Unknown"

    def check_consistency(self, mappings: List[MappingCandidate]) -> List[MappingCandidate]:
        logger.info("Running Semantic Logical Consistency Check and Reasoning Rules...")
        
        # We will apply three key rules:
        # Rule 1: Disjointness Check (ESG Dimension Match) -> Reject/Penalize mismatches
        # Rule 2: Taxonomic/Structural Propagation -> Strengthen if parent/ancestor concepts match
        # Rule 3: Property Consistency Verification -> Validate/boost if units & datatypes are compatible, penalize if conflicting
        
        repaired_mappings = []
        
        for m in mappings:
            score = m.similarity_score
            m.reasoning = []
            
            # Rule 1: Disjointness Check (Reject/Penalize)
            brsr_cat = self.classify_concept_esg(m.brsr_concept)
            gri_cat = self.classify_concept_esg(m.gri_concept)
            
            if brsr_cat != "Unknown" and gri_cat != "Unknown":
                if brsr_cat != gri_cat:
                    logger.warning(
                        f"⚠️ Disjointness violation: BRSR '{m.brsr_concept.label}' ({brsr_cat}) "
                        f"mapped to GRI '{m.gri_concept.label}' ({gri_cat}). Penalizing score."
                    )
                    score *= 0.1  # Severe penalty, effectively rejecting the mapping
                    m.reasoning.append(f"Rule 1 (Disjointness Check Violation): BRSR category '{brsr_cat}' and GRI category '{gri_cat}' conflict. Applied disjointness score penalty (scaled by 0.1).")
                else:
                    m.reasoning.append(f"Rule 1 (Disjointness Check): Both concepts verified as ESG dimension '{brsr_cat}' (No disjointness violation).")
            else:
                m.reasoning.append(f"Rule 1 (Disjointness Check): Skipped (unknown category for BRSR or GRI concept).")
                
            # Rule 2: Taxonomic Propagation (Strengthen)
            # If parents have matching terms, strengthen the sub-concept mapping
            s_path = m.brsr_concept.hierarchy_path
            t_path = m.gri_concept.hierarchy_path
            if s_path and t_path:
                # Compare direct parent labels using simple word overlap
                parent_sim = self._string_sim(s_path[0], t_path[0])
                if parent_sim >= 0.5:
                    logger.info(
                        f"📈 Strengthening mapping via parent alignment: '{s_path[0]}' <-> '{t_path[0]}' (sim={parent_sim:.2f})"
                    )
                    score = min(1.0, score * 1.15)
                    m.reasoning.append(f"Rule 2 (Taxonomic Propagation): Alignment boosted by 1.15x due to high similarity of parent concepts ('{s_path[0]}' <-> '{t_path[0]}' at {parent_sim:.2f}).")
                    
            # Rule 3: Property Consistency (Validate / Penalize)
            # Check if datatype and unit are compatible or conflicting
            s_dt = (m.brsr_concept.datatype or "").lower()
            t_dt = (m.gri_concept.datatype or "").lower()
            s_is_num = any(w in s_dt for w in ["int", "decimal", "float", "double", "number"])
            t_is_num = any(w in t_dt for w in ["int", "decimal", "float", "double", "number"])
            
            # If one is numeric and the other is textual, penalize the mapping
            if (s_dt and t_dt) and (s_is_num != t_is_num):
                logger.warning(
                    f"⚠️ Property mismatch: BRSR '{m.brsr_concept.label}' (Numeric: {s_is_num}) "
                    f"vs GRI '{m.gri_concept.label}' (Numeric: {t_is_num}). Penalizing mapping."
                )
                score *= 0.8
                m.reasoning.append(f"Rule 3 (Property Mismatch): Mismatched datatypes (BRSR Numeric: {s_is_num}, GRI Numeric: {t_is_num}). Penalized score by 0.8x.")
            # If both are numeric and have matching units, boost
            elif s_is_num and t_is_num and m.brsr_concept.unit and m.gri_concept.unit:
                if m.brsr_concept.unit.lower().strip() == m.gri_concept.unit.lower().strip():
                    score = min(1.0, score * 1.05)
                    m.reasoning.append(f"Rule 3 (Property Consistency): Perfect unit match ('{m.brsr_concept.unit}'). Boosted score by 1.05x.")
                else:
                    m.reasoning.append(f"Rule 3 (Property Consistency): Both numeric, different units ('{m.brsr_concept.unit}' vs '{m.gri_concept.unit}'). No penalty.")

            m.similarity_score = float(score)
            repaired_mappings.append(m)
            
        # 4. Incoherence Repair: Enforce 1-to-1 equivalence mappings
        repaired_mappings.sort(key=lambda x: x.similarity_score, reverse=True)
        
        final_repaired = []
        used_brsr_eq = set()
        used_gri_eq = set()
        
        for m in repaired_mappings:
            brsr_uri = m.brsr_concept.uri
            gri_uri = m.gri_concept.uri
            
            if m.similarity_score >= 0.85:
                if brsr_uri in used_brsr_eq or gri_uri in used_gri_eq:
                    m.similarity_score = min(m.similarity_score, 0.74)
                    m.reasoning.append("Rule 4 (Mapping Repair): Enforced 1-to-1 equivalence restriction. Downgraded concept mapping to 0.74 (Non-Equivalent) due to existing higher-confidence match.")
                else:
                    used_brsr_eq.add(brsr_uri)
                    used_gri_eq.add(gri_uri)
                    m.reasoning.append("Rule 4 (Mapping Repair): Retained 1-to-1 equivalence mapping.")
            
            final_repaired.append(m)
            
        return final_repaired

    def _string_sim(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        w1 = set(re.findall(r'\w+', s1.lower()))
        w2 = set(re.findall(r'\w+', s2.lower()))
        inter = w1.intersection(w2)
        union = w1.union(w2)
        return len(inter) / len(union) if union else 0.0
