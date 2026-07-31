import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from confidence.learner import ConfidenceLearner

logger = logging.getLogger(__name__)

class ConfidenceAggregator:
    """Confidence Aggregator for EXTENSION_3.
    Aggregates similarity scores using automatically learned feature weights from ground-truth BRSR-ESRS candidate alignments.
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
        logger.info(f"Aggregating matcher confidence scores using learned feature weights: {self.weights}...")
        aggregated = {}
        
        reasoning_scores = reasoning_scores or {}
        embedding_scores = embedding_scores or {}
        
        all_keys = set(lexical_scores.keys()) | set(structural_scores.keys()) | set(property_scores.keys()) | set(reasoning_scores.keys()) | set(embedding_scores.keys())
        
        w_lex = self.weights.get("lexical", 0.35)
        w_struc = self.weights.get("structural", 0.30)
        w_prop = self.weights.get("property", 0.15)
        w_emb = self.weights.get("embedding", self.weights.get("reasoning", 0.20))
        
        total_w = w_lex + w_struc + w_prop + w_emb
        if total_w <= 0:
            total_w = 1.0
            
        w_lex /= total_w
        w_struc /= total_w
        w_prop /= total_w
        w_emb /= total_w
        
        for key in all_keys:
            lex = lexical_scores.get(key, 0.0)
            struc = structural_scores.get(key, 0.0)
            prop = property_scores.get(key, 0.5)
            emb = embedding_scores.get(key, reasoning_scores.get(key, 0.0))
            
            score = (lex * w_lex + struc * w_struc + prop * w_prop + emb * w_emb)
            
            if lex < 0.15 and emb < 0.15:
                score = 0.0
                
            aggregated[key] = float(score)
            
        return aggregated
