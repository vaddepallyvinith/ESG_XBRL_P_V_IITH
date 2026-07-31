"""
learner.py - Automatic Confidence Weight Learning Module for BRSR-GRI Alignment.
Automatically learns optimal feature weights for [Lexical, Structural, Property, Reasoning]
using Grid Search optimization / Logistic Regression to maximize Precision, Recall, and F1.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

LEARNED_WEIGHTS_FILE = Path(__file__).parent / "learned_weights.json"

class ConfidenceLearner:
    """Automatic Confidence Learning Module."""
    
    def __init__(self, weights_file: Optional[Path] = None):
        self.weights_file = Path(weights_file or LEARNED_WEIGHTS_FILE)
        
    def learn_weights(
        self,
        features: List[Dict[str, float]],
        labels: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Learns optimal weights automatically from feature vectors [L, S, P, R].
        
        Args:
            features: List of dicts with keys 'lexical', 'structural', 'property', 'reasoning'.
            labels: Binary labels (1 for true match, 0 for non-match). If None, pseudo-labels are generated.
            
        Returns:
            Dict of learned weights normalizing to 1.0.
        """
        logger.info("🤖 Executing Automatic Confidence Weight Optimization...")
        
        if not features:
            logger.warning("No feature vectors provided. Using default fallback weights.")
            default_w = {"lexical": 0.40, "structural": 0.35, "property": 0.15, "reasoning": 0.10}
            self.save_weights(default_w)
            return default_w

        # If no explicit labels are provided, construct high-precision pseudo-ground truth
        if labels is None or len(labels) != len(features):
            labels = []
            for f in features:
                # Target alignment candidate indicator: Lexical >= 0.12 or Structural >= 0.18
                if f.get("lexical", 0.0) >= 0.12 or f.get("structural", 0.0) >= 0.18:
                    labels.append(1)
                else:
                    labels.append(0)

        # Priority 1: Try Logistic Regression if sufficient positives exist
        y = np.array(labels)
        num_positives = np.sum(y == 1)
        
        best_weights = None
        
        if num_positives >= 5:
            try:
                from sklearn.linear_model import LogisticRegression
                X = np.array([
                    [f.get("lexical", 0.0), f.get("structural", 0.0), f.get("property", 0.0), f.get("reasoning", 0.0)]
                    for f in features
                ])
                clf = LogisticRegression(fit_intercept=False, max_iter=1000)
                clf.fit(X, y)
                coefs = np.maximum(clf.coef_[0], 0.01)  # Ensure non-negative weights
                norm_coefs = coefs / np.sum(coefs)
                best_weights = {
                    "lexical": float(round(norm_coefs[0], 4)),
                    "structural": float(round(norm_coefs[1], 4)),
                    "property": float(round(norm_coefs[2], 4)),
                    "reasoning": float(round(norm_coefs[3], 4)),
                }
                logger.info(f"✅ Logistic Regression Weight Optimization Succeeded: {best_weights}")
            except Exception as e:
                logger.warning(f"Logistic Regression optimization fallback to Grid Search: {e}")

        # Priority 2: Grid Search optimization over weight grid if LR unavailable or failed
        if best_weights is None:
            best_f1 = -1.0
            best_weights = {"lexical": 0.40, "structural": 0.35, "property": 0.15, "reasoning": 0.10}
            
            # Grid search over normalized weight combinations
            grid_steps = np.linspace(0.05, 0.60, 12)
            
            for w_l in grid_steps:
                for w_s in grid_steps:
                    for w_p in [0.05, 0.10, 0.15, 0.20]:
                        for w_r in [0.05, 0.10, 0.15, 0.20]:
                            total_w = w_l + w_s + w_p + w_r
                            norm_l, norm_s, norm_p, norm_r = w_l/total_w, w_s/total_w, w_p/total_w, w_r/total_w
                            
                            # Calculate candidate confidence scores
                            scores = np.array([
                                f.get("lexical", 0.0) * norm_l +
                                f.get("structural", 0.0) * norm_s +
                                f.get("property", 0.0) * norm_p +
                                f.get("reasoning", 0.0) * norm_r
                                for f in features
                            ])
                            
                            preds = (scores >= 0.55).astype(int)
                            
                            tp = np.sum((preds == 1) & (y == 1))
                            fp = np.sum((preds == 1) & (y == 0))
                            fn = np.sum((preds == 0) & (y == 1))
                            
                            prec = tp / max(1, tp + fp)
                            rec = tp / max(1, tp + fn)
                            f1 = (2 * prec * rec) / max(1e-6, prec + rec)
                            
                            if f1 > best_f1:
                                best_f1 = f1
                                best_weights = {
                                    "lexical": float(round(norm_l, 4)),
                                    "structural": float(round(norm_s, 4)),
                                    "property": float(round(norm_p, 4)),
                                    "reasoning": float(round(norm_r, 4)),
                                }
            logger.info(f"✅ Grid Search Weight Optimization Succeeded (Best F1={best_f1:.4f}): {best_weights}")

        self.save_weights(best_weights)
        return best_weights

    def save_weights(self, weights: Dict[str, float]):
        """Save learned weights to JSON."""
        self.weights_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.weights_file, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=4)
        logger.info(f"💾 Saved learned weights to {self.weights_file}")

    def load_weights(self) -> Dict[str, float]:
        """Load learned weights from JSON."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, "r", encoding="utf-8") as f:
                    weights = json.load(f)
                logger.info(f"📖 Loaded learned weights from {self.weights_file}: {weights}")
                return weights
            except Exception as e:
                logger.warning(f"Failed to load learned weights from {self.weights_file}: {e}")
        
        # Fallback default
        return {"lexical": 0.40, "structural": 0.35, "property": 0.15, "reasoning": 0.10}
