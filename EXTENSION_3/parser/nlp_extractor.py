import os
import json
import logging
from typing import List, Any
import concurrent.futures

import google.generativeai as genai
from parser.section_segmenter import DocumentTree, GRIDocumentTree, DisclosureNode, GRIRequirementNode, GRIDisclosureNode

logger = logging.getLogger(__name__)

class EnrichmentEngine:
    """Enriches extracted ESG nodes with Metric, Unit, Datatype, and Applicability using Gemini LLM."""
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment. Skipping NLP extraction.")
            self.enabled = False
            return
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.enabled = True

    def enrich_trees(self, trees: List[Any]):
        """Enrich a batch of document trees."""
        if not self.enabled:
            return
            
        logger.info("Starting NLP enrichment (Metric, Unit, Datatype, Applicability)...")
        
        nodes_to_process = []
        
        # Gather all nodes to process
        for tree in trees:
            if isinstance(tree, DocumentTree):
                for section in tree.sections:
                    for principle in section.principles:
                        for ig in principle.indicator_groups:
                            for disc in ig.disclosures:
                                nodes_to_process.append(disc)
            elif isinstance(tree, GRIDocumentTree):
                for std in tree.standards:
                    for disc in std.disclosures:
                        nodes_to_process.append(disc)
                        for req in disc.requirements:
                            nodes_to_process.append(req)
                            
        logger.info(f"Found {len(nodes_to_process)} nodes to enrich. Calling Gemini API (Parallel)...")
        
        # Parallel execution to speed up API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(self._process_node, nodes_to_process))
            
        logger.info("NLP enrichment complete.")

    def _process_node(self, node: Any):
        """Send a single node to Gemini for extraction."""
        # Extract text context from node
        text_parts = []
        if hasattr(node, "text") and node.text:
            text_parts.append(node.text)
        if hasattr(node, "content_blocks") and node.content_blocks:
            text_parts.append(" ".join(node.content_blocks))
            
        text_to_analyze = " ".join(text_parts).strip()
        
        if not text_to_analyze or len(text_to_analyze) < 15:
            return # Skip very short/empty nodes
            
        prompt = f"""
        Extract the following information from the given ESG reporting requirement text.
        Text: '{text_to_analyze[:1500]}'
        
        Respond ONLY with a valid JSON object matching this schema exactly, with no markdown code blocks around it:
        {{
            "metric": "string or null (e.g., Scope 1 emissions)",
            "unit": "string or null (e.g., Metric tons CO2e)",
            "datatype": "string (Quantitative, Qualitative, or Boolean)",
            "applicability": "string (Mandatory, Recommendation, or Optional)"
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            
            # Clean up markdown if LLM returned it
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            data = json.loads(raw_text)
            
            # Update Node
            if hasattr(node, "metric"):
                node.metric = data.get("metric")
            if hasattr(node, "unit"):
                node.unit = data.get("unit")
            if hasattr(node, "datatype"):
                node.datatype = data.get("datatype")
            if hasattr(node, "applicability"):
                node.applicability = data.get("applicability")
                
        except Exception as e:
            # Silently catch rate limit / parsing errors for individual nodes so batch continues
            pass
