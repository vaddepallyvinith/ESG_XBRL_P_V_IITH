import os
import json
import logging
import pickle
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from mapping.models import OntologyConcept, MappingEvidence, MappingCandidate, FinalMapping
from mapping.llm_verifier import LLMVerifier

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    
logger = logging.getLogger(__name__)

RSO = Namespace("http://example.org/ontology/rso#")
SCHEMA = Namespace("http://schema.org/")

class SemanticMappingEngine:
    def __init__(self, config: dict):
        self.config = config.get("mapping", {})
        self.embedding_model_name = self.config.get("embedding_model", "all-mpnet-base-v2")
        
        # Load batch size for LLM verifications
        llm_config = config.get("llm", {})
        self.llm_batch_size = llm_config.get("batch_size", 10)
        self.llm_verifier = LLMVerifier(model_name=llm_config.get("model", "meta-llama/llama-3.3-70b-instruct"))
        
        if ST_AVAILABLE:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)
        else:
            logger.warning("sentence-transformers not installed. Embedding similarity will be 0.")
            self.embedder = None
            
        self.graph = Graph()
        self.brsr_concepts: List[OntologyConcept] = []
        self.gri_concepts: List[OntologyConcept] = []
        
        self.weights = self.config.get("weights", {
            "definition_similarity": 0.25,
            "embedding_similarity": 0.20,
            "hierarchy_similarity": 0.15,
            "relationship_similarity": 0.10,
            "topic_similarity": 0.10,
            "label_similarity": 0.10,
            "unit_compatibility": 0.05,
            "datatype_compatibility": 0.05,
            "context_similarity": 0.05,
            "graph_similarity": 0.10 # New graph topology weight
        })
        self.thresholds = self.config.get("thresholds", {
            "equivalent": 0.90,
            "partial": 0.75,
            "broader_narrower": 0.55
        })

    def run(self, ontology_path: str, output_dir: str):
        logger.info(f"Step 1: Loading ontology from {ontology_path}")
        self.graph.parse(ontology_path, format="turtle")
        
        logger.info("Step 2: Extracting BRSR and GRI concepts")
        self._extract_concepts()
        
        logger.info(f"Found {len(self.brsr_concepts)} BRSR concepts and {len(self.gri_concepts)} GRI concepts.")
        
        logger.info("Step 3 & 4: Generating candidates and collecting evidence")
        # Pass output_dir for embedding cache
        candidates = self._generate_and_evaluate_candidates(output_dir)
        
        logger.info(f"Step 8 & 9: Verifying top {min(100, len(candidates))} candidates with LLM and scoring confidence")
        final_mappings = self._verify_and_score(candidates[:100])
        
        logger.info("Step 10: Exporting Mapping Repository")
        self._export_results(final_mappings, output_dir)
        
    def _extract_concepts(self):
        # Find Disclosures and Requirements
        for concept_class in [RSO.Disclosure, RSO.Requirement]:
            for s in self.graph.subjects(RDF.type, concept_class):
                concept = self._build_concept(s, concept_class)
                if concept:
                    if concept.framework == "BRSR":
                        self.brsr_concepts.append(concept)
                    elif concept.framework == "GRI":
                        self.gri_concepts.append(concept)

    def _build_concept(self, uri: URIRef, concept_class: URIRef) -> OntologyConcept:
        uri_str = str(uri)
        
        source_doc = self.graph.value(uri, RSO.sourceDocument)
        source_doc_str = str(source_doc) if source_doc else ""
        
        if "BRSR" in source_doc_str or "Annexure" in source_doc_str or "Q" in uri_str.split("_")[-2:]:
            framework = "BRSR"
        elif "GRI" in source_doc_str or "GRI" in uri_str:
            framework = "GRI"
        elif "Q" in uri_str:
            framework = "BRSR"
        else:
            return None
            
        label = self.graph.value(uri, SCHEMA.name)
        text = self.graph.value(uri, SCHEMA.text)
        datatype = self.graph.value(uri, RSO.hasDatatype)
        applicability = self.graph.value(uri, RSO.hasApplicability)
        
        metric_name = None
        for m in self.graph.objects(uri, RSO.hasMetric):
            metric_name = str(self.graph.value(m, SCHEMA.name) or "")
            
        unit_name = None
        for u in self.graph.objects(uri, RSO.hasUnit):
            unit_name = str(self.graph.value(u, SCHEMA.name) or "")
            
        # Get hierarchy (belongsTo chain) for structural topology
        hierarchy = []
        current = uri
        for _ in range(5): # up to 5 levels
            parent = self.graph.value(current, RSO.belongsTo)
            if parent:
                parent_label = self.graph.value(parent, SCHEMA.name)
                if parent_label:
                    hierarchy.append(str(parent_label))
                current = parent
            else:
                break
                
        return OntologyConcept(
            uri=uri_str,
            framework=framework,
            label=str(label) if label else "",
            concept_type="Requirement" if concept_class == RSO.Requirement else "Disclosure",
            definition=str(text) if text else "",
            metric=metric_name,
            unit=unit_name,
            datatype=str(datatype) if datatype else None,
            applicability=str(applicability) if applicability else None,
            hierarchy_path=hierarchy
        )

    def _get_cached_embeddings(self, texts: List[str], cache_path: str) -> np.ndarray:
        """Embed texts using SentenceTransformer with disk caching."""
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")

        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache
        for i, text in enumerate(texts):
            if text in cache:
                embeddings.append(cache[text])
            else:
                embeddings.append(None) # placeholder
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                
        # Encode missing
        if texts_to_encode:
            logger.info(f"Encoding {len(texts_to_encode)} new texts (not in cache)...")
            new_embeddings = self.embedder.encode(texts_to_encode, convert_to_tensor=False)
            
            # Fill placeholders and update cache
            for idx, emb, text in zip(indices_to_encode, new_embeddings, texts_to_encode):
                embeddings[idx] = emb
                cache[text] = emb
                
            # Save cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache, f)
            except Exception as e:
                logger.warning(f"Failed to save embedding cache: {e}")

        return np.array(embeddings)

    def _generate_and_evaluate_candidates(self, output_dir: str) -> List[MappingCandidate]:
        candidates = []
        cache_path = os.path.join(output_dir, "embeddings_cache.pkl")
        
        if self.embedder and self.brsr_concepts and self.gri_concepts:
            brsr_texts = [c.label + " " + c.definition for c in self.brsr_concepts]
            gri_texts = [c.label + " " + c.definition for c in self.gri_concepts]
            
            logger.info("Retrieving BRSR embeddings...")
            brsr_emb = self._get_cached_embeddings(brsr_texts, cache_path)
            logger.info("Retrieving GRI embeddings...")
            gri_emb = self._get_cached_embeddings(gri_texts, cache_path)
            
            # Convert back to torch tensors for fast cosine sim
            import torch
            brsr_tensor = torch.tensor(brsr_emb)
            gri_tensor = torch.tensor(gri_emb)
            
            cosine_scores = util.cos_sim(brsr_tensor, gri_tensor).cpu().numpy()
        else:
            cosine_scores = np.zeros((len(self.brsr_concepts), len(self.gri_concepts)))

        top_k = self.config.get("candidate_top_k", 10)
        
        for i, brsr in enumerate(self.brsr_concepts):
            top_indices = np.argsort(cosine_scores[i])[-top_k:][::-1]
            
            for j in top_indices:
                gri = self.gri_concepts[j]
                
                # Weak candidate pre-filtering: skip if cosine score is abysmally low
                if cosine_scores[i][j] < 0.2:
                    continue
                    
                evidence = MappingEvidence()
                evidence.embedding_similarity = float(cosine_scores[i][j])
                
                b_words = set(brsr.label.lower().split())
                g_words = set(gri.label.lower().split())
                if b_words and g_words:
                    evidence.label_similarity = len(b_words & g_words) / len(b_words | g_words)
                    
                if brsr.unit and gri.unit:
                    evidence.unit_compatibility = 1.0 if brsr.unit.lower() == gri.unit.lower() else 0.2
                else:
                    evidence.unit_compatibility = 0.5
                    
                if brsr.datatype and gri.datatype:
                    evidence.datatype_compatibility = 1.0 if brsr.datatype.lower() == gri.datatype.lower() else 0.0
                else:
                    evidence.datatype_compatibility = 0.5
                    
                b_hier = set(brsr.hierarchy_path)
                g_hier = set(gri.hierarchy_path)
                if b_hier and g_hier:
                    b_hier_words = set(" ".join(b_hier).lower().split())
                    g_hier_words = set(" ".join(g_hier).lower().split())
                    evidence.hierarchy_similarity = len(b_hier_words & g_hier_words) / max(1, len(b_hier_words | g_hier_words))
                
                # Topological graph similarity approximation (difference in hierarchy depth)
                depth_diff = abs(len(brsr.hierarchy_path) - len(gri.hierarchy_path))
                # 0 diff -> 1.0 sim, 1 diff -> 0.8, 2 diff -> 0.6, etc.
                graph_sim = max(0.0, 1.0 - (depth_diff * 0.2))
                
                evidence.relationship_similarity = 0.5
                evidence.topic_similarity = 0.5
                evidence.context_similarity = 0.5
                
                score = (
                    evidence.embedding_similarity * self.weights.get("embedding_similarity", 0.20) +
                    evidence.embedding_similarity * self.weights.get("definition_similarity", 0.25) +
                    evidence.label_similarity * self.weights.get("label_similarity", 0.10) +
                    evidence.unit_compatibility * self.weights.get("unit_compatibility", 0.05) +
                    evidence.datatype_compatibility * self.weights.get("datatype_compatibility", 0.05) +
                    evidence.hierarchy_similarity * self.weights.get("hierarchy_similarity", 0.15) +
                    evidence.relationship_similarity * self.weights.get("relationship_similarity", 0.10) +
                    evidence.topic_similarity * self.weights.get("topic_similarity", 0.10) +
                    evidence.context_similarity * self.weights.get("context_similarity", 0.05) +
                    graph_sim * self.weights.get("graph_similarity", 0.10)
                )
                
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    score = score / total_weight

                if score >= self.thresholds["broader_narrower"]:
                    candidates.append(MappingCandidate(
                        brsr_concept=brsr,
                        gri_concept=gri,
                        evidence=evidence,
                        similarity_score=score
                    ))
                    
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates

    def _verify_and_score(self, candidates: List[MappingCandidate]) -> List[FinalMapping]:
        final_mappings = []
        batch_size = self.llm_batch_size
        
        # Process in batches
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            logger.info(f"LLM Verifying batch {i//batch_size + 1}/{(len(candidates) + batch_size - 1)//batch_size}...")
            
            results = self.llm_verifier.verify_mappings_batch(batch)
            
            for cand, (decision, explanation) in zip(batch, results):
                if cand.similarity_score >= self.thresholds["equivalent"]:
                    relationship = "Equivalent"
                elif cand.similarity_score >= self.thresholds["partial"]:
                    relationship = "Partial Equivalent"
                elif cand.similarity_score >= self.thresholds["broader_narrower"]:
                    relationship = "Broader/Narrower"
                else:
                    relationship = "NotMapped"
                    
                if decision == "Disagree":
                    relationship = "NotMapped"
                    
                conf = cand.similarity_score * 100
                if decision == "Agree":
                    conf = min(100.0, conf + 15.0)
                elif decision == "Disagree":
                    conf = max(0.0, conf - 40.0)
                    
                final_mappings.append(FinalMapping(
                    brsr_uri=cand.brsr_concept.uri,
                    gri_uri=cand.gri_concept.uri,
                    brsr_label=cand.brsr_concept.label,
                    gri_label=cand.gri_concept.label,
                    relationship=relationship,
                    confidence_score=conf,
                    similarity_score=cand.similarity_score,
                    evidence_summary=cand.evidence.model_dump(),
                    llm_verification=decision,
                    llm_explanation=explanation
                ))
                
        return final_mappings
        
    def _export_results(self, mappings: List[FinalMapping], output_dir: str):
        out_path = os.path.join(output_dir, "mapping_repository.json")
        with open(out_path, "w") as f:
            json.dump([m.model_dump() for m in mappings], f, indent=2)
            
        csv_path = os.path.join(output_dir, "mapping_summary.csv")
        df = pd.DataFrame([m.model_dump() for m in mappings])
        if not df.empty:
            df.to_csv(csv_path, index=False)
            
        logger.info(f"Exported {len(mappings)} mappings to {out_path} and {csv_path}")
