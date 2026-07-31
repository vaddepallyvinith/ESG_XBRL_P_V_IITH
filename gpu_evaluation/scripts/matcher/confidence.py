import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class ConfidenceAggregator:
    """Confidence Aggregator.
    Aggregates similarity scores from different matchers using weighted heuristics.
    """
    def __init__(self, config: dict):
        self.config = config.get("confidence", {})
        self.weights = self.config.get("weights", {
            "lexical": 0.40,
            "structural": 0.35,
            "property": 0.15,
            "embedding": 0.10
        })

    def aggregate(self, 
                  lexical_scores: Dict[Tuple[str, str], float], 
                  structural_scores: Dict[Tuple[str, str], float], 
                  property_scores: Dict[Tuple[str, str], float],
                  embedding_scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        logger.info("Aggregating matcher confidence scores...")
        aggregated = {}
        
        all_keys = set(lexical_scores.keys()) | set(structural_scores.keys()) | set(property_scores.keys()) | set(embedding_scores.keys())
        
        for key in all_keys:
            lex = lexical_scores.get(key, 0.0)
            struc = structural_scores.get(key, 0.0)
            prop = property_scores.get(key, 0.5)
            emb = embedding_scores.get(key, 0.0)
            
            w_lex = self.weights.get("lexical", 0.35)
            w_struc = self.weights.get("structural", 0.30)
            w_prop = self.weights.get("property", 0.15)
            w_emb = self.weights.get("embedding", 0.20)
            
            if key not in property_scores:
                w_prop = 0.0
                
            total_w = w_lex + w_struc + w_prop + w_emb
            
            score = (lex * w_lex + struc * w_struc + prop * w_prop + emb * w_emb)
            if total_w > 0:
                score /= total_w
                
            # If both lexical and embedding similarities are extremely low, clamp overall score to 0
            if lex < 0.15 and emb < 0.15:
                score = 0.0
                
            aggregated[key] = float(score)
            
        return aggregated
