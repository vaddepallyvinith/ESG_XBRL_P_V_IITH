import os
import json
import logging
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class MappingEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.template_dir = "templates"
        from mapping.multi_llm_evaluator import MultiLLMEvaluator
        self.multi_evaluator = MultiLLMEvaluator(config)
        
    def run_multi_llm_evaluation(self, input_dir: str):
        repo_path = os.path.join(input_dir, "mapping_repository.json")
        out_path = os.path.join(input_dir, "multi_llm_results.json")
        
        if not os.path.exists(repo_path):
            logger.error(f"Mapping repository not found at {repo_path}")
            return
            
        # Check if already evaluated to avoid redundant costs
        if os.path.exists(out_path):
            logger.info(f"Multi-LLM results already exist at {out_path}. Skipping API calls.")
            return
            
        with open(repo_path, "r") as f:
            mappings = json.load(f)
            
        logger.info("Running Multi-LLM Evaluation...")
        results = self.multi_evaluator.evaluate_mappings(mappings)
        
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved multi-LLM evaluation results to {out_path}")
        
    def _plot_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str
        
    def generate_dashboard(self, input_dir: str, output_dir: str):
        repo_path = os.path.join(input_dir, "mapping_repository.json")
        if not os.path.exists(repo_path):
            logger.error(f"Mapping repository not found at {repo_path}")
            return
            
        with open(repo_path, "r") as f:
            mappings = json.load(f)
            
        if not mappings:
            logger.warning("Mapping repository is empty.")
            
        df = pd.DataFrame(mappings)
        
        # Aggregate stats
        total_mappings = len(mappings)
        relationships = Counter([m.get("relationship", "Unknown") for m in mappings])
        verifications = Counter([m.get("llm_verification", "Unknown") for m in mappings])
        
        # Generate Plots
        plt.switch_backend('Agg')
        
        # 1. Relationship Distribution
        fig_rel, ax_rel = plt.subplots(figsize=(6, 4))
        if not df.empty and 'relationship' in df.columns:
            rel_counts = df['relationship'].value_counts()
            ax_rel.pie(rel_counts, labels=rel_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4C51BF', '#48BB78', '#ED8936', '#F56565'])
            ax_rel.axis('equal')
        rel_chart = self._plot_to_base64(fig_rel)
        
        # 2. Confidence Distribution
        fig_conf, ax_conf = plt.subplots(figsize=(6, 4))
        if not df.empty and 'confidence_score' in df.columns:
            ax_conf.hist(df['confidence_score'], bins=10, color='#4299E1', edgecolor='black')
            ax_conf.set_title("Confidence Score Distribution")
            ax_conf.set_xlabel("Confidence (%)")
            ax_conf.set_ylabel("Count")
        conf_chart = self._plot_to_base64(fig_conf)
        
        # Render HTML
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            
        # Ensure template exists
        template_path = os.path.join(self.template_dir, "dashboard.html")
        if not os.path.exists(template_path):
            logger.error(f"Template not found at {template_path}")
            return
            
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template("dashboard.html")
        
        html_content = template.render(
            total_mappings=total_mappings,
            relationships=dict(relationships),
            verifications=dict(verifications),
            rel_chart=rel_chart,
            conf_chart=conf_chart,
            mappings=mappings
        )
        
        out_path = os.path.join(output_dir, "mapping_dashboard.html")
        with open(out_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Generated dashboard at {out_path}")
        
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
            
            # For agreement rate calculation, we assume base model or majority vote? 
            # Or just agreement % with the generated mappings. The prompt asks for "Verification agreement rate"
            # which is % of "Agree" decisions.
            for model_name, metrics in multi_res.items():
                decisions = metrics.get("decisions", [])
                agrees = sum(1 for d in decisions if d == "Agree")
                agreement_rate = (agrees / len(decisions) * 100) if decisions else 0.0
                
                avg_time = metrics.get("avg_response_time", 0.0)
                cost_100 = metrics.get("cost_per_100_mappings", 0.0)
                
                print(f"{model_name:<20} | {agreement_rate:<14.2f}% | {avg_time:<15.2f} | ${cost_100:<14.4f}")
                
        print("--------------------------------\n")
