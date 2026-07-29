import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from matcher.models import OntologyConcept, MappingEvidence, MappingCandidate, FinalMapping
from verifier.llm_verifier import LLMVerifier

from matcher.lexical_matcher import LexicalMatcher
from matcher.structural_matcher import StructuralMatcher
from matcher.property_matcher import PropertyMatcher
from matcher.confidence import ConfidenceAggregator
from matcher.ontology_reasoner import OntologyReasoner
from matcher.skos_mapper import SKOSMapper

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
            "graph_similarity": 0.10
        })
        self.thresholds = self.config.get("thresholds", {
            "equivalent": 0.90,
            "partial": 0.75,
            "broader_narrower": 0.55
        })

        # Initialize AML Matchers
        self.lexical_matcher = LexicalMatcher(config)
        self.structural_matcher = StructuralMatcher(config)
        self.property_matcher = PropertyMatcher(config)
        self.confidence_aggregator = ConfidenceAggregator(config)
        self.ontology_reasoner = OntologyReasoner(config)
        self.skos_mapper = SKOSMapper(config)

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
                    elif concept.framework in ("ESRS", "GRI"):
                        self.gri_concepts.append(concept)

    def _build_concept(self, uri: URIRef, concept_class: URIRef) -> OntologyConcept:
        uri_str = str(uri)
        
        source_doc = self.graph.value(uri, RSO.sourceDocument)
        source_doc_str = str(source_doc) if source_doc else ""
        
        if "BRSR" in source_doc_str or "Annexure" in source_doc_str or "Q" in uri_str.split("_")[-2:]:
            framework = "BRSR"
        elif "ESRS" in source_doc_str or "ESRS" in uri_str or "Delegated" in source_doc_str:
            framework = "ESRS"
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
            # Query specific subproperties first, falling back to belongsTo
            parent = (self.graph.value(current, RSO.belongsToTopic) or
                      self.graph.value(current, RSO.belongsToFramework) or
                      self.graph.value(current, RSO.belongsToDisclosure) or
                      self.graph.value(current, RSO.belongsTo))
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
        
        # 1. Run primary matchers
        lexical_scores = self.lexical_matcher.match(self.brsr_concepts, self.gri_concepts)
        structural_scores = self.structural_matcher.match(self.brsr_concepts, self.gri_concepts)
        property_scores = self.property_matcher.match(self.brsr_concepts, self.gri_concepts)
        
        # 2. Candidate Selection
        candidate_pairs = []
        for brsr in self.brsr_concepts:
            for target in self.gri_concepts:
                lex_val = lexical_scores.get((brsr.uri, target.uri), 0.0)
                struc_val = structural_scores.get((brsr.uri, target.uri), 0.0)
                if lex_val >= 0.08 or struc_val >= 0.20:
                    candidate_pairs.append((brsr, target))
                    
        logger.info(f"Ontology-guided candidate selection: selected {len(candidate_pairs)} candidate pairs.")
        
        # 3. Retrieve embeddings if available
        cosine_scores = np.zeros((len(self.brsr_concepts), len(self.gri_concepts)))
        if self.embedder and self.brsr_concepts and self.gri_concepts:
            brsr_texts = [c.label + " " + c.definition for c in self.brsr_concepts]
            target_texts = [c.label + " " + c.definition for c in self.gri_concepts]
            
            brsr_emb = self._get_cached_embeddings(brsr_texts, cache_path)
            target_emb = self._get_cached_embeddings(target_texts, cache_path)
            
            import torch
            brsr_tensor = torch.tensor(brsr_emb)
            target_tensor = torch.tensor(target_emb)
            cosine_scores = util.cos_sim(brsr_tensor, target_tensor).cpu().numpy()
            
        brsr_to_idx = {c.uri: idx for idx, c in enumerate(self.brsr_concepts)}
        target_to_idx = {c.uri: idx for idx, c in enumerate(self.gri_concepts)}
        
        embedding_scores = {}
        for brsr, target in candidate_pairs:
            bi = brsr_to_idx[brsr.uri]
            gi = target_to_idx[target.uri]
            emb_val = float(cosine_scores[bi][gi])
            lex_val = lexical_scores.get((brsr.uri, target.uri), 0.0)
            if lex_val < 0.05:
                embedding_scores[(brsr.uri, target.uri)] = emb_val * 0.3
            else:
                embedding_scores[(brsr.uri, target.uri)] = emb_val

        # 4. Filter scoring dictionaries to candidate pairs & extract independent feature vectors [L, S, P, R]
        filtered_lexical = {}
        filtered_structural = {}
        filtered_property = {}
        filtered_embedding = {}
        reasoning_scores = {}
        features_list = []
        
        for brsr, target in candidate_pairs:
            key = (brsr.uri, target.uri)
            l_val = lexical_scores.get(key, 0.0)
            s_val = structural_scores.get(key, 0.0)
            p_val = property_scores.get(key, 0.5)
            r_val = 0.5 if brsr.concept_type == target.concept_type else 0.2
            if l_val > 0.3 and s_val > 0.3:
                r_val += 0.3
            r_val = min(1.0, r_val)
            
            filtered_lexical[key] = l_val
            filtered_structural[key] = s_val
            filtered_property[key] = p_val
            filtered_embedding[key] = embedding_scores.get(key, 0.0)
            reasoning_scores[key] = r_val
            
            features_list.append({
                "lexical": l_val,
                "structural": s_val,
                "property": p_val,
                "reasoning": r_val
            })

        # 5. Automatically learn weights via ConfidenceLearner
        from confidence.learner import ConfidenceLearner
        learner = ConfidenceLearner()
        self.learned_weights = learner.learn_weights(features_list)
        self.confidence_aggregator.set_learned_weights(self.learned_weights)
        
        # 6. Aggregate matching evidence using learned weight model
        aggregated_scores = self.confidence_aggregator.aggregate(
            lexical_scores=filtered_lexical,
            structural_scores=filtered_structural,
            property_scores=filtered_property,
            reasoning_scores=reasoning_scores,
            embedding_scores=filtered_embedding
        )
        
        # 7. Generate candidate objects
        for brsr, target in candidate_pairs:
            key = (brsr.uri, target.uri)
            score = aggregated_scores.get(key, 0.0)
            
            if score < 0.20: # Keep candidate candidates
                continue
                
            evidence = MappingEvidence()
            evidence.embedding_similarity = filtered_embedding.get(key, 0.0)
            evidence.label_similarity = filtered_lexical.get(key, 0.0)
            evidence.hierarchy_similarity = filtered_structural.get(key, 0.0)
            
            prop_val = filtered_property.get(key, 0.5)
            evidence.datatype_compatibility = prop_val
            evidence.unit_compatibility = prop_val
            evidence.relationship_similarity = reasoning_scores.get(key, 0.5)
            evidence.topic_similarity = filtered_structural.get(key, 0.0)
            evidence.context_similarity = filtered_lexical.get(key, 0.0)
            
            candidates.append(MappingCandidate(
                brsr_concept=brsr,
                gri_concept=target,
                evidence=evidence,
                similarity_score=score
            ))
            
        candidates = self.ontology_reasoner.check_consistency(candidates)
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates

    def _verify_and_score(self, candidates: List[MappingCandidate]) -> List[FinalMapping]:
        final_mappings = []
        
        logger.info(f"Scoring and mapping {len(candidates)} candidates using learned confidence model...")
        for cand in candidates:
            if cand.similarity_score >= self.thresholds["equivalent"]:
                relationship = "Equivalent"
            elif cand.similarity_score >= self.thresholds["partial"]:
                relationship = "Partial Equivalent"
            elif cand.similarity_score >= self.thresholds["broader_narrower"]:
                b_type = cand.brsr_concept.concept_type
                g_type = cand.gri_concept.concept_type
                if b_type == "Disclosure" and g_type == "Requirement":
                    relationship = "Broader"
                elif b_type == "Requirement" and g_type == "Disclosure":
                    relationship = "Narrower"
                else:
                    if len(cand.brsr_concept.definition) > len(cand.gri_concept.definition):
                        relationship = "Broader"
                    else:
                        relationship = "Narrower"
            else:
                relationship = "NotMapped"
                
            conf = cand.similarity_score * 100
            
            b_id = cand.brsr_concept.uri.split("#")[-1]
            e_id = cand.gri_concept.uri.split("#")[-1]
            
            final_mappings.append(FinalMapping(
                brsr_uri=cand.brsr_concept.uri,
                gri_uri=cand.gri_concept.uri,
                brsr_label=cand.brsr_concept.label,
                gri_label=cand.gri_concept.label,
                relationship=relationship,
                confidence_score=conf,
                similarity_score=cand.similarity_score,
                evidence_summary=cand.evidence.model_dump(),
                llm_verification="Pending",
                llm_explanation="Phase 3 SKOS alignment complete (LLM verification pending)",
                ontology_path="",
                brsr_id=b_id,
                gri_id=e_id,
                lexical_score=cand.evidence.label_similarity,
                structural_score=cand.evidence.hierarchy_similarity,
                property_score=cand.evidence.datatype_compatibility,
                reasoning_score=cand.evidence.relationship_similarity,
                overall_confidence=conf,
                skos_relation="",
                reasoning=[f"Learned model confidence score: {conf:.1f}%"]
            ))
            
        # Apply SKOS Mapper
        final_mappings = self.skos_mapper.map_to_skos(final_mappings)
        return final_mappings
        
    def _export_results(self, mappings: List[FinalMapping], output_dir: str):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export mapping.json & mapping_repository.json
        mapping_data = [m.model_dump() for m in mappings]
        with open(out_dir / "mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)
        with open(out_dir / "mapping_repository.json", "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)
            
        # 2. Export mapping.csv & mapping_summary.csv
        df = pd.DataFrame(mapping_data)
        if not df.empty:
            df.to_csv(out_dir / "mapping.csv", index=False)
            df.to_csv(out_dir / "mapping_summary.csv", index=False)
            
        # 3. Export learned_weights.json
        weights_data = getattr(self, "learned_weights", {"lexical": 0.40, "structural": 0.35, "property": 0.15, "reasoning": 0.10})
        with open(out_dir / "learned_weights.json", "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=4)
        with open(Path(__file__).parent.parent / "confidence" / "learned_weights.json", "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=4)

        # 4. Export confidence_report.json
        conf_values = [m.similarity_score for m in mappings]
        high_conf = sum(1 for c in conf_values if c >= 0.85)
        med_conf = sum(1 for c in conf_values if 0.70 <= c < 0.85)
        low_conf = sum(1 for c in conf_values if 0.55 <= c < 0.70)
        
        conf_report = {
            "learned_weights": weights_data,
            "total_candidates_evaluated": len(mappings),
            "confidence_distribution": {
                "high_confidence_0.85_1.0": high_conf,
                "medium_confidence_0.70_0.85": med_conf,
                "low_confidence_0.55_0.70": low_conf,
                "average_confidence_score": float(np.mean(conf_values)) if conf_values else 0.0
            },
            "skos_relation_counts": {
                "exactMatch": sum(1 for m in mappings if "exactMatch" in str(m.ontology_path)),
                "closeMatch": sum(1 for m in mappings if "closeMatch" in str(m.ontology_path)),
                "broadMatch": sum(1 for m in mappings if "broadMatch" in str(m.ontology_path)),
                "narrowMatch": sum(1 for m in mappings if "narrowMatch" in str(m.ontology_path)),
                "relatedMatch": sum(1 for m in mappings if "relatedMatch" in str(m.ontology_path)),
            }
        }
        with open(out_dir / "confidence_report.json", "w", encoding="utf-8") as f:
            json.dump(conf_report, f, indent=2)

        # 5. Export mapping.ttl (SKOS RDF Graph)
        skos_g = Graph()
        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
        skos_g.bind("skos", SKOS)
        
        for m in mappings:
            b_uri = URIRef(m.brsr_uri)
            e_uri = URIRef(m.gri_uri)
            rel_uri = URIRef(m.ontology_path) if m.ontology_path else SKOS.relatedMatch
            skos_g.add((b_uri, rel_uri, e_uri))
            
        skos_g.serialize(destination=str(out_dir / "mapping.ttl"), format="turtle")
            
        logger.info(f"✅ Successfully exported all mapping artifacts & reports to {out_dir}")
