import os
import json
import logging
from collections import Counter
import pandas as pd

logger = logging.getLogger(__name__)

class MappingEvaluator:
    def __init__(self, config: dict):
        self.config = config
        from evaluation.multi_llm_evaluator import MultiLLMEvaluator
        self.multi_evaluator = MultiLLMEvaluator(config)
        
    def run_multi_llm_evaluation(self, input_dir: str):
        repo_path = os.path.join(input_dir, "mapping_repository.json")
        out_path = os.path.join(input_dir, "multi_llm_results.json")
        
        if not os.path.exists(repo_path):
            logger.error(f"Mapping repository not found at {repo_path}")
            return
            
        with open(repo_path, "r") as f:
            mappings = json.load(f)
            
        logger.info("Running Multi-LLM Evaluation pipeline...")
        self.multi_evaluator.evaluate_mappings(mappings, checkpoint_path=out_path)
        
    def calculate_comparative_metrics(self, mappings, multi_res):
        # Align decisions
        groq_decs = multi_res.get("Groq", {}).get("decisions", [])
        if not groq_decs or len(groq_decs) == 0:
            return None
        
        # Dynamically find all other models in multi_res that have non-empty decisions
        other_models = []
        for m_name in multi_res.keys():
            if m_name == "Groq":
                continue
            decs = multi_res[m_name].get("decisions", [])
            if len(decs) > 0:
                other_models.append(m_name)
        
        # Determine alignment length
        n = min(len(mappings), len(groq_decs))
        for m_name in other_models:
            n = min(n, len(multi_res[m_name].get("decisions", [])))
            
        if n == 0:
            return None
            
        aligned_mappings = mappings[:n]
        groq_decs = groq_decs[:n]
        
        def encode(x):
            return 1 if x == "Agree" else 0
            
        groq_bin = [encode(x) for x in groq_decs]
        
        # Build correlation dict starting with Groq (GT)
        df_data = {"Groq (GT)": groq_bin}
        
        # Align other models
        model_bins = {}
        for m_name in other_models:
            decs = multi_res[m_name].get("decisions", [])[:n]
            m_bin = [encode(x) for x in decs]
            model_bins[m_name] = m_bin
            df_data[m_name] = m_bin
            
        # Heuristics: "Partial Equivalent" / "Equivalent" -> 1, else 0
        heuristic_decs = []
        # Base LLM from mapping_repository
        base_llm_decs = []
        for m in aligned_mappings:
            rel = m.get("relationship", "")
            if rel in ["Partial Equivalent", "Equivalent"]:
                heuristic_decs.append("Agree")
            else:
                heuristic_decs.append("Disagree")
            base_llm_decs.append(m.get("llm_verification", "Disagree"))
            
        heuristic_bin = [encode(x) for x in heuristic_decs]
        base_llm_bin = [encode(x) for x in base_llm_decs]
        
        df_data["Heuristic"] = heuristic_bin
        df_data["Base LLM"] = base_llm_bin
        
        df_corr = pd.DataFrame(df_data)
        corr_matrix = df_corr.corr(method='pearson').fillna(0.0)
        
        # Calculate metrics for each model relative to Groq (GT)
        def get_stats(y_true, y_pred):
            import numpy as np
            yt = np.array(y_true)
            yp = np.array(y_pred)
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            tn = np.sum((yt == 0) & (yp == 0))
            
            acc = (tp + tn) / len(y_true)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
            }
            
        metrics = {}
        for m_name in other_models:
            metrics[m_name] = get_stats(groq_bin, model_bins[m_name])
            
        metrics["Heuristic (Ontology)"] = get_stats(groq_bin, heuristic_bin)
        metrics["Base LLM (Phase 3)"] = get_stats(groq_bin, base_llm_bin)
        
        return {
            "metrics": metrics,
            "corr_matrix": corr_matrix.to_dict(),
            "aligned_count": n
        }
        
    def generate_cli_report(self, input_dir: str):
        repo_path = os.path.join(input_dir, "mapping_repository.json")
        if not os.path.exists(repo_path):
            return
            
        with open(repo_path, "r") as f:
            mappings = json.load(f)
            
        if not mappings:
            print("\n📊 --- Mapping Summary Report ---")
            print("No mappings found.")
            return
            
        df = pd.DataFrame(mappings)
        print("\n📊 --- Mapping Summary Report ---")
        print(f"Total Correspondences Found: {len(df)}")
        print("\nDistribution by Relationship Type:")
        print(df['relationship'].value_counts().to_string())
        
        if 'confidence_score' in df.columns:
            print(f"\nAverage Confidence Score: {df['confidence_score'].mean():.2f}%")
            
        if 'llm_verification' in df.columns:
            print("\nLLM Verification Audit (Base Model):")
            print(df['llm_verification'].value_counts().to_string())
            
        # Multi-LLM results
        multi_out_path = os.path.join(input_dir, "multi_llm_results.json")
        if os.path.exists(multi_out_path):
            with open(multi_out_path, "r") as f:
                multi_res = json.load(f)
                
            print("\n🤖 --- Multi-LLM Comparative Evaluation ---")
            print(f"{'Model Name':<20} | {'Agreement %':<15} | {'Avg Time (s)':<15} | {'Cost/100 ($)':<15}")
            print("-" * 75)
            
            for model_name, metrics in multi_res.items():
                decisions = metrics.get("decisions", [])
                agrees = sum(1 for d in decisions if d == "Agree")
                agreement_rate = (agrees / len(decisions) * 100) if decisions else 0.0
                
                avg_time = metrics.get("avg_response_time", 0.0)
                cost_100 = metrics.get("cost_per_100_mappings", 0.0)
                
                print(f"{model_name:<20} | {agreement_rate:<14.2f}% | {avg_time:<15.2f} | ${cost_100:<14.4f}")
            
            # Print Comparative Stats (Groq as Ground Truth)
            comp_metrics = self.calculate_comparative_metrics(mappings, multi_res)
            if comp_metrics:
                print("\n🎯 --- Comparative Performance Metrics (Ground Truth = Groq) ---")
                print(f"{'Model/Source':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
                print("-" * 75)
                for model_name, m in comp_metrics["metrics"].items():
                    print(f"{model_name:<25} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
                
                print("\n📈 --- Pearson Correlation Matrix (Agree=1, Disagree=0) ---")
                corr = comp_metrics["corr_matrix"]
                models = list(corr.keys())
                print(f"{'':<20}", end="")
                for m in models:
                    print(f"| {m[:12]:<12}", end="")
                print("\n" + "-" * 85)
                for m1 in models:
                    print(f"{m1:<20}", end="")
                    for m2 in models:
                        print(f"| {corr[m1][m2]:<12.4f}", end="")
                    print()
                
        print("--------------------------------\n")
