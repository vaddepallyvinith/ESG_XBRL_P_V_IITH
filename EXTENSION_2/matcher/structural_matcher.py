import logging
from typing import List, Dict, Tuple
from matcher.models import OntologyConcept

logger = logging.getLogger(__name__)

import logging
import re
from typing import List, Dict, Tuple
from matcher.models import OntologyConcept

logger = logging.getLogger(__name__)

class StructuralMatcher:
    """AML-inspired Structural/Taxonomic Matcher.
    Leverages hierarchical relationships, parent-sibling-ancestor similarity, and topology.
    """
    def __init__(self, config: dict):
        self.config = config.get("structural_matcher", {})

    def _string_sim(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        w1 = set(re.findall(r'\w+', s1.lower()))
        w2 = set(re.findall(r'\w+', s2.lower()))
        inter = w1.intersection(w2)
        union = w1.union(w2)
        return len(inter) / len(union) if union else 0.0

    def match(self, source_concepts: List[OntologyConcept], target_concepts: List[OntologyConcept]) -> Dict[Tuple[str, str], float]:
        logger.info("Running Structural Matcher...")
        scores = {}
        
        for s_concept in source_concepts:
            s_path = s_concept.hierarchy_path
            for t_concept in target_concepts:
                t_path = t_concept.hierarchy_path
                
                if not s_path and not t_path:
                    scores[(s_concept.uri, t_concept.uri)] = 0.0
                    continue
                
                # 1. Direct Parent Similarity
                parent_sim = 0.0
                if s_path and t_path:
                    parent_sim = self._string_sim(s_path[0], t_path[0])
                
                # 2. Ancestor Path similarity with distance-decay weights
                weighted_intersection = 0.0
                weighted_union = 0.0
                
                s_path_lower = [p.lower() for p in s_path]
                t_path_lower = [p.lower() for p in t_path]
                
                for idx_s, s_anc in enumerate(s_path_lower):
                    w_s = 1.0 / (idx_s + 1.0)
                    best_match_sim = 0.0
                    best_idx_t = 0
                    for idx_t, t_anc in enumerate(t_path_lower):
                        sim = self._string_sim(s_anc, t_anc)
                        if sim > best_match_sim:
                            best_match_sim = sim
                            best_idx_t = idx_t
                            
                    w_t = 1.0 / (best_idx_t + 1.0)
                    combined_weight = (w_s + w_t) / 2.0
                    weighted_intersection += best_match_sim * combined_weight
                    weighted_union += combined_weight
                
                for idx_t, t_anc in enumerate(t_path_lower):
                    w_t = 1.0 / (idx_t + 1.0)
                    best_match_sim = 0.0
                    for idx_s, s_anc in enumerate(s_path_lower):
                        sim = self._string_sim(s_anc, t_anc)
                        if sim > best_match_sim:
                            best_match_sim = sim
                    if best_match_sim < 0.5:
                        weighted_union += w_t
                        
                path_sim = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
                
                # 3. Depth Difference Penalty
                depth_diff = abs(len(s_path) - len(t_path))
                depth_factor = max(0.0, 1.0 - (depth_diff * 0.15))
                
                scores[(s_concept.uri, t_concept.uri)] = float((parent_sim * 0.4) + (path_sim * 0.4) + (depth_factor * 0.2))
                
        return scores
