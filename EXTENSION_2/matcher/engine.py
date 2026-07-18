import os
import json
import logging
import pickle
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
        
        # 1. Run AML primary matchers on the entire concepts list (highly efficient HashMap/dictionary searches)
        lexical_scores = self.lexical_matcher.match(self.brsr_concepts, self.gri_concepts)
        structural_scores = self.structural_matcher.match(self.brsr_concepts, self.gri_concepts)
        property_scores = self.property_matcher.match(self.brsr_concepts, self.gri_concepts)
        
        # 2. Ontology-Guided Candidate Selection:
        # Generate candidates based *primarily* on ontology evidence (lexical overlap or taxonomic alignment)
        candidate_pairs = []
        for brsr in self.brsr_concepts:
            for gri in self.gri_concepts:
                lex_val = lexical_scores.get((brsr.uri, gri.uri), 0.0)
                struc_val = structural_scores.get((brsr.uri, gri.uri), 0.0)
                
                # Rule: Candidate must have some lexical overlap OR significant taxonomic alignment
                if lex_val >= 0.12 or struc_val >= 0.35:
                    candidate_pairs.append((brsr, gri))
                    
        logger.info(f"Ontology-guided candidate selection: selected {len(candidate_pairs)} candidate pairs.")
        
        # 3. Retrieve embeddings for the concepts if available
        cosine_scores = np.zeros((len(self.brsr_concepts), len(self.gri_concepts)))
        if self.embedder and self.brsr_concepts and self.gri_concepts:
            brsr_texts = [c.label + " " + c.definition for c in self.brsr_concepts]
            gri_texts = [c.label + " " + c.definition for c in self.gri_concepts]
            
            logger.info("Retrieving BRSR embeddings...")
            brsr_emb = self._get_cached_embeddings(brsr_texts, cache_path)
            logger.info("Retrieving GRI embeddings...")
            gri_emb = self._get_cached_embeddings(gri_texts, cache_path)
            
            import torch
            brsr_tensor = torch.tensor(brsr_emb)
            gri_tensor = torch.tensor(gri_emb)
            cosine_scores = util.cos_sim(brsr_tensor, gri_tensor).cpu().numpy()
            
        # Build index mapping for easy lookup
        brsr_to_idx = {c.uri: idx for idx, c in enumerate(self.brsr_concepts)}
        gri_to_idx = {c.uri: idx for idx, c in enumerate(self.gri_concepts)}
        
        # 4. Populate embedding scores specifically for the ontology-selected candidate pairs
        # Embeddings act as supporting evidence only.
        embedding_scores = {}
        for brsr, gri in candidate_pairs:
            bi = brsr_to_idx[brsr.uri]
            gi = gri_to_idx[gri.uri]
            emb_val = float(cosine_scores[bi][gi])
            
            # If there is zero lexical overlap, heavily penalize the embedding's influence to prevent false positives
            lex_val = lexical_scores.get((brsr.uri, gri.uri), 0.0)
            if lex_val < 0.05:
                embedding_scores[(brsr.uri, gri.uri)] = emb_val * 0.3
            else:
                embedding_scores[(brsr.uri, gri.uri)] = emb_val

        # 5. Filter all scoring dictionaries to candidate pairs only before aggregation
        filtered_lexical = {}
        filtered_structural = {}
        filtered_property = {}
        filtered_embedding = {}
        
        for brsr, gri in candidate_pairs:
            key = (brsr.uri, gri.uri)
            filtered_lexical[key] = lexical_scores.get(key, 0.0)
            filtered_structural[key] = structural_scores.get(key, 0.0)
            filtered_property[key] = property_scores.get(key, 0.0)
            filtered_embedding[key] = embedding_scores.get(key, 0.0)
            
        # 6. Aggregate matching evidence scores with the ConfidenceAggregator
        aggregated_scores = self.confidence_aggregator.aggregate(
            lexical_scores=filtered_lexical,
            structural_scores=filtered_structural,
            property_scores=filtered_property,
            embedding_scores=filtered_embedding
        )
        
        # 7. Generate candidates list filtered by thresholds
        for brsr, gri in candidate_pairs:
            key = (brsr.uri, gri.uri)
            score = aggregated_scores.get(key, 0.0)
            
            # Minimum threshold filter
            if score < self.thresholds["broader_narrower"]:
                continue
                
            evidence = MappingEvidence()
            evidence.embedding_similarity = filtered_embedding.get(key, 0.0)
            evidence.label_similarity = filtered_lexical.get(key, 0.0)
            evidence.hierarchy_similarity = filtered_structural.get(key, 0.0)
            
            prop_val = filtered_property.get(key, 0.5)
            evidence.datatype_compatibility = prop_val
            evidence.unit_compatibility = prop_val
            
            evidence.relationship_similarity = 0.5
            evidence.topic_similarity = 0.5
            evidence.context_similarity = 0.5
            
            candidates.append(MappingCandidate(
                brsr_concept=brsr,
                gri_concept=gri,
                evidence=evidence,
                similarity_score=score
            ))
            
        # 8. Apply Ontology Reasoner (disjointness, taxonomic propagation, property consistency, and mapping repair)
        candidates = self.ontology_reasoner.check_consistency(candidates)
        
        # 6. Sort by resolved similarity score
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates

    def _verify_and_score(self, candidates: List[MappingCandidate]) -> List[FinalMapping]:
        final_mappings = []
        
        logger.info(f"Scoring and mapping {len(candidates)} candidates using ontology evidence...")
        for cand in candidates:
            if cand.similarity_score >= self.thresholds["equivalent"]:
                relationship = "Equivalent"
            elif cand.similarity_score >= self.thresholds["partial"]:
                relationship = "Partial Equivalent"
            elif cand.similarity_score >= self.thresholds["broader_narrower"]:
                # Determine broader vs narrower based on concept types
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
            
            final_mappings.append(FinalMapping(
                brsr_uri=cand.brsr_concept.uri,
                gri_uri=cand.gri_concept.uri,
                brsr_label=cand.brsr_concept.label,
                gri_label=cand.gri_concept.label,
                relationship=relationship,
                confidence_score=conf,
                similarity_score=cand.similarity_score,
                evidence_summary=cand.evidence.model_dump(),
                llm_verification="Skipped",
                llm_explanation="LLM verification skipped",
                ontology_path=""
            ))
            
        # Apply SKOS Mapper to convert relationships to standard SKOS properties
        final_mappings = self.skos_mapper.map_to_skos(final_mappings)
        
        # Prepare rich verification payload for LLM verifier
        verification_payload = []
        for cand, fm in zip(candidates, final_mappings):
            skos_rel = fm.ontology_path.split("#")[-1] if fm.ontology_path else "relatedMatch"
            verification_payload.append({
                "brsr_id": cand.brsr_concept.uri.split("#")[-1],
                "brsr_label": cand.brsr_concept.label,
                "brsr_definition": cand.brsr_concept.definition,
                "brsr_hierarchy": cand.brsr_concept.hierarchy_path,
                "gri_id": cand.gri_concept.uri.split("#")[-1],
                "gri_label": cand.gri_concept.label,
                "gri_definition": cand.gri_concept.definition,
                "gri_hierarchy": cand.gri_concept.hierarchy_path,
                "lexical_score": cand.evidence.label_similarity,
                "structural_score": cand.evidence.hierarchy_similarity,
                "property_score": cand.evidence.datatype_compatibility,
                "reasoning_score": cand.similarity_score,
                "overall_confidence": cand.similarity_score * 100,
                "skos_relation": skos_rel,
                "evidence_summary": cand.evidence.model_dump(),
                "reasoning": cand.reasoning
            })
            
        logger.info(f"Invoking LLM verification batch for {len(verification_payload)} candidate mappings...")
        verification_results = self.llm_verifier.verify_mappings_rich_batch(verification_payload)
        
        # Update mappings with LLM verification decision and fill new requested keys
        for cand, fm, v_res, v_pay in zip(candidates, final_mappings, verification_results, verification_payload):
            fm.llm_verification = v_res["verification"]
            fm.llm_explanation = v_res["explanation"]
            
            # Fill new requested fields
            fm.brsr_id = v_pay["brsr_id"]
            fm.gri_id = v_pay["gri_id"]
            fm.lexical_score = v_pay["lexical_score"]
            fm.structural_score = v_pay["structural_score"]
            fm.property_score = v_pay["property_score"]
            fm.reasoning_score = v_pay["reasoning_score"]
            fm.overall_confidence = v_pay["overall_confidence"]
            fm.skos_relation = v_pay["skos_relation"]
            fm.reasoning = cand.reasoning
            fm.verification = v_res
            
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
