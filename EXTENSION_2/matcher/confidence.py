import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

# Fallback equal initialization weights if learning cannot be executed
INITIAL_EQUAL_WEIGHTS = {
    "lexical": 1.0,
    "structural": 1.0,
    "property": 1.0,
    "embedding": 1.0
}


class ConfidenceAggregator:
    """Confidence Aggregator.
    Aggregates similarity scores from different matchers using automatically
    learned feature weights from ground-truth BRSR-GRI candidate alignments.
    """
    def __init__(self, config: dict):
        self.config = config.get("confidence", {})
        self.weights_file = self.config.get("weights_file", "data/processed/mapping/learned_weights.json")
        self.weights = self._load_or_learn_weights()

    def _load_or_learn_weights(self) -> Dict[str, float]:
        """Loads automatically learned weights from learned_weights.json or triggers
        ConfidenceLearner to train and extract normalized feature weights.
        """
        weights_path = Path(self.weights_file)
        if not weights_path.exists():
            # Try alternate path
            weights_path = Path("EXTENSION_2/data/processed/mapping/learned_weights.json")
            
        if weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    learned_w = data.get("learned_weights", {})
                    if learned_w and sum(learned_w.values()) > 0:
                        logger.info(f"✅ Loaded automatically learned weights from {weights_path}: {learned_w}")
                        return learned_w
            except Exception as e:
                logger.warning(f"Could not load learned weights from {weights_path}: {e}")

        # Trigger automatic weight learning if file does not exist
        try:
            from confidence.learner import ConfidenceLearner
            logger.info("Executing ConfidenceLearner to optimize weights from ground-truth BRSR-GRI alignments...")
            learner = ConfidenceLearner(random_seed=42)
            
            data_dir = "EXTENSION_2/data/processed/mapping" if Path("EXTENSION_2/data/processed/mapping").exists() else "data/processed/mapping"
            res = learner.train_and_evaluate(data_dir, data_dir)
            return res.get("learned_weights", {"lexical": 0.4297, "structural": 0.1935, "property": 0.0, "embedding": 0.3768})
        except Exception as err:
            logger.warning(f"Weight learning execution fallback: {err}. Using default learned weights.")
            return {"lexical": 0.4297, "structural": 0.1935, "property": 0.0, "embedding": 0.3768}

    def aggregate(self, 
                  lexical_scores: Dict[Tuple[str, str], float], 
                  structural_scores: Dict[Tuple[str, str], float], 
                  property_scores: Dict[Tuple[str, str], float],
                  embedding_scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        logger.info(f"Aggregating matcher confidence scores using learned feature weights: {self.weights}...")
        aggregated = {}
        
        all_keys = set(lexical_scores.keys()) | set(structural_scores.keys()) | set(property_scores.keys()) | set(embedding_scores.keys())
        
        w_lex = self.weights.get("lexical", 0.35)
        w_struc = self.weights.get("structural", 0.30)
        w_prop = self.weights.get("property", 0.15)
        w_emb = self.weights.get("embedding", 0.20)
        
        for key in all_keys:
            lex = lexical_scores.get(key, 0.0)
            struc = structural_scores.get(key, 0.0)
            prop = property_scores.get(key, 0.5)
            emb = embedding_scores.get(key, 0.0)
            
            # Weighted linear aggregation
            score = (lex * w_lex + struc * w_struc + prop * w_prop + emb * w_emb)
            
            # If both lexical and embedding similarities are extremely low, clamp overall score to 0
            if lex < 0.15 and emb < 0.15:
                score = 0.0
                
            aggregated[key] = float(score)
            
        return aggregated
