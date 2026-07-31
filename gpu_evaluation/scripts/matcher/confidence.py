import logging
from typing import Dict, Tuple, Optional, List
from confidence.learner import ConfidenceLearner

logger = logging.getLogger(__name__)

class ConfidenceAggregator:
    """Confidence Aggregator.
    Aggregates similarity scores using automatically learned confidence weights.
    Inference Pipeline:
      Similarity Features [L, S, P, R] → Learned Weight Model → Base Confidence → Reasoning Adjustment → Final Confidence
    """
    def __init__(self, config: dict):
        self.config = config.get("confidence", {})
        self.learner = ConfidenceLearner()
        self.weights = self.learner.load_weights()

    def set_learned_weights(self, weights: Dict[str, float]):
        """Sets active learned weights dynamically."""
        self.weights = weights

    def aggregate(
        self, 
        lexical_scores: Dict[Tuple[str, str], float], 
        structural_scores: Dict[Tuple[str, str], float], 
        property_scores: Dict[Tuple[str, str], float],
        reasoning_scores: Optional[Dict[Tuple[str, str], float]] = None,
        embedding_scores: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[Tuple[str, str], float]:
        logger.info(f"Aggregating matcher confidence scores using learned weights: {self.weights}")
        aggregated = {}
        
        reasoning_scores = reasoning_scores or {}
        embedding_scores = embedding_scores or {}
        
        all_keys = set(lexical_scores.keys()) | set(structural_scores.keys()) | set(property_scores.keys()) | set(reasoning_scores.keys()) | set(embedding_scores.keys())
        
        w_lex = self.weights.get("lexical", 0.40)
        w_struc = self.weights.get("structural", 0.35)
        w_prop = self.weights.get("property", 0.15)
        w_reas = self.weights.get("reasoning", 0.10)
        
        total_w = w_lex + w_struc + w_prop + w_reas
        if total_w <= 0:
            total_w = 1.0
            
        w_lex /= total_w
        w_struc /= total_w
        w_prop /= total_w
        w_reas /= total_w
        
        for key in all_keys:
            lex = lexical_scores.get(key, 0.0)
            struc = structural_scores.get(key, 0.0)
            prop = property_scores.get(key, 0.5)
            reas = reasoning_scores.get(key, 0.0)
            emb = embedding_scores.get(key, 0.0)
            
            # Base confidence from learned weights model
            base_confidence = (lex * w_lex + struc * w_struc + prop * w_prop + reas * w_reas)
            
            # Base confidence boost if embedding similarity is also available
            if emb > 0.0 and emb > lex:
                base_confidence = 0.8 * base_confidence + 0.2 * emb

            # Apply ontology reasoning adjustment constraints
            final_confidence = base_confidence
            if lex < 0.12 and struc < 0.25 and emb < 0.20:
                final_confidence = 0.0
                
            aggregated[key] = float(np.clip(final_confidence, 0.0, 1.0) if 'np' in globals() else min(1.0, max(0.0, final_confidence)))
            
        return aggregated
