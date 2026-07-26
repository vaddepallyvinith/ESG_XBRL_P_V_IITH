# BRSR-GRI Semantic Mapping Framework: System Architecture & Technical Specifications

This document outlines the end-to-end 5-stage architecture of the **BRSR-GRI Ontology-Guided Semantic Mapping Engine**, detailing the system components, module implementations, code structures, and operational commands.

---

## 🏗️ System Architecture Flowchart

```text
┌──────────────────────────┐
│ 1. ESG Disclosure Input  │  PDF Parsing & Document Tree Structuring
│    (BRSR / GRI Reports)  │  (PyMuPDF fitz, Layout Boundary Detection, NLP)
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ 2. Ontology Construction │  OWL 2 DL Ontology & RDF Turtle Graph Generation
│  • Concept Extraction    │  (RDFLib, Class Hierarchies, Neo4j Graph CSVs)
│  • ESG Ontology Creation │
└─────────────┬────────────┘
              │
              ▼
┌────────────────────────────────────┐
│ 3. Ontology-Guided Matching        │  AML-Inspired Deterministic Alignment Engine
│  • Lexical Similarity              │  (Jaccard, Taxonomic Path Decay, Unit Compatibility,
│  • Structural Similarity           │   SentenceTransformers, Disjointness Rules)
│  • Property Similarity             │
│  • Ontology Reasoning              │
└─────────────┬──────────────────────┘
              │
              ▼
┌──────────────────────────┐
│ 4. Alignment Generation  │  Confidence Aggregation & SKOS Standardization
│  • Confidence Aggregation│  (skos:exactMatch, skos:closeMatch,
│  • SKOS Mapping          │   skos:broadMatch, skos:narrowMatch)
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ 5. Validation & Output   │  Multi-LLM Benchmarking & Audit Reports
│  • LLM Verification      │  (Post-Hoc Verification, CoT Explanations,
│  • Explanation           │   Precision/Recall/F1, Pearson Matrix)
│  • Evaluation & Reports  │
└──────────────────────────┘
```

---

## 1. ESG Disclosure Input (BRSR / GRI Reports)

```text
┌──────────────────────────┐
│ 1. ESG Disclosure Input  │
│    (BRSR / GRI Reports)  │
└──────────────────────────┘
```

### Module Specifications & Code References
* **PDF Parser**: [`parser/pdf_parser.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/parser/pdf_parser.py) (`parse_raw_directory()`, `PDFParser.parse()`)
* **Heading Detector**: [`parser/heading_detector.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/parser/heading_detector.py) (`HeadingDetector.detect()`)
* **Section Segmenter**: [`parser/section_segmenter.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/parser/section_segmenter.py) (`SectionSegmenter.segment()`, `GRISegmenter.segment()`)
* **NLP Enrichment**: [`parser/nlp_extractor.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/parser/nlp_extractor.py) (`EnrichmentEngine.enrich_trees()`)
* **JSON Exporter**: [`parser/json_exporter.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/parser/json_exporter.py) (`export_batch()`)

### Code Implementation (`main.py`)
```python
# Parse raw PDFs into structured document objects
documents = parse_raw_directory(raw_dir, company)

# Segment into hierarchical document trees
brsr_segmenter = SectionSegmenter()
gri_segmenter = GRISegmenter()

for doc in documents:
    if "GRI" in doc.file_name:
        tree = gri_segmenter.segment(doc)
    else:
        tree = brsr_segmenter.segment(doc)

# NLP Enrichment & JSON export
enricher = EnrichmentEngine()
enricher.enrich_trees(trees)
export_batch(trees, output_dir)
```

### Technical Highlights
1. **Multi-Source Ingestion**: Ingests raw PDFs for 30 BRSR company annual reports and the complete GRI 300-series Environmental Standards.
2. **Hierarchy Preservation**: Uses PyMuPDF (`fitz`) and bounding-box layout analysis to extract hierarchical structures: $\text{Sections} \rightarrow \text{Disclosures} \rightarrow \text{Requirements} \rightarrow \text{Tables} \rightarrow \text{Metrics}$.
3. **NLP Entity Extraction**: Extracts measurement units ($tCO_2e$, $GJ$, $KL$), numeric datatypes, and reporting applicability.

---

## 2. Ontology Construction

```text
┌──────────────────────────┐
│ 2. Ontology Construction │
│  • Concept Extraction    │
│  • ESG Ontology Creation │
└──────────────────────────┘
```

### Module Specifications & Code References
* **Schema Definition**: [`ontology/schema.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/ontology/schema.py) (`create_base_graph()`)
  * Classes: `rso:Disclosure`, `rso:Requirement`, `rso:Metric`, `rso:Unit`
  * Object Properties: `rso:belongsToTopic`, `rso:contains`, `rso:hasUnit`
* **Graph Builder**: [`ontology/builder.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/ontology/builder.py) (`OntologyBuilder.build()`, `_parse_brsr()`, `_parse_gri()`, `_add_node()`, `_add_edge()`)

### Code Implementation (`ontology/schema.py` & `ontology/builder.py`)
```python
# Define OWL Classes & Properties (schema.py)
g.add((RSO.Disclosure, RDF.type, OWL.Class))
g.add((RSO.Requirement, RDF.type, OWL.Class))
g.add((RSO.belongsToTopic, RDF.type, OWL.ObjectProperty))

# Populate RDF Triples (builder.py)
self.graph.add((disc_uri, RDF.type, RSO.Disclosure))
self.graph.add((disc_uri, SCHEMA.name, Literal(disclosure_name)))
self.graph.add((disc_uri, RSO.belongsToTopic, topic_uri))

# Serialize to Turtle Graph (.ttl)
graph.serialize(destination="data/processed/ontology/esg_ontology.ttl", format="turtle")
```

### Technical Highlights
1. **Formal Web Ontology (OWL 2 DL)**: Formalizes BRSR and GRI framework structures into two ontologies (**BRSR-EO** and **GRI-EO**) using `RDFLib`.
2. **RDF Serialization**: Exports nodes and relationships to an RDF Turtle graph (`esg_ontology.ttl`, ~887 triples) under namespace `http://example.org/ontology/rso#`.
3. **Neo4j Graph Database Export**: Simultaneously generates `neo4j_nodes.csv` (1,356 nodes) and `neo4j_relationships.csv` (2,624 edges) for graph database population.

---

## 3. Ontology-Guided Matching

```text
┌────────────────────────────────────┐
│ 3. Ontology-Guided Matching        │
│  • Lexical Similarity              │
│  • Structural Similarity           │
│  • Property Similarity             │
│  • Ontology Reasoning              │
└────────────────────────────────────┘
```

### Module Specifications & Code References
* **Lexical Matcher**: [`matcher/lexical_matcher.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/lexical_matcher.py) (`LexicalMatcher.calculate_similarity()`)
* **Structural Matcher**: [`matcher/structural_matcher.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/structural_matcher.py) (`StructuralMatcher.calculate_similarity()`)
* **Property Matcher**: [`matcher/property_matcher.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/property_matcher.py) (`PropertyMatcher.calculate_similarity()`)
* **Ontology Reasoner**: [`matcher/ontology_reasoner.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/ontology_reasoner.py) (`OntologyReasoner.evaluate_rules()`)
* **Semantic Engine**: [`matcher/engine.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/engine.py) (`SemanticMappingEngine._generate_and_evaluate_candidates()`)

### Code Implementation (`matcher/engine.py`)
```python
# Aggregate multi-evidence similarity signals
S_ont = (
    w_lex * sim_lex + 
    w_struc * sim_struc + 
    w_prop * sim_prop + 
    w_emb * sim_emb
)

# Apply Active Ontology Reasoner Rules (disjointness & propagation)
penalty_or_boost = reasoner.evaluate_rules(concept_brsr, concept_gri)
S_final = S_ont * penalty_or_boost
```

### Technical Highlights
1. **AgreementMakerLight (AML) Inspired**: Operates deterministically on ontological evidence rather than unconstrained LLM prompts.
2. **Multi-Evidence Similarity Signals**:
   * **Lexical**: Jaccard token overlap on concept labels.
   * **Structural**: Taxonomic path decay $e^{-\lambda \cdot d(c_1, c_2)}$ over parent/child trees.
   * **Property**: Data type and unit compatibility verification ($tCO_2e$ vs $GJ$).
   * **Dense Embeddings**: Cosine similarity via `SentenceTransformers` (`all-mpnet-base-v2`).
3. **Ontology Reasoner**: Applies active logical consistency rules (e.g., applies a $0.1\times$ penalty for domain disjointness violations like *Water* vs *GHG Emissions*, or a $1.15\times$ boost for aligned parent topics).

---

## 4. Alignment Generation

```text
┌──────────────────────────┐
│ 4. Alignment Generation  │
│  • Confidence Aggregation│
│  • SKOS Mapping          │
│    - exactMatch          │
│    - closeMatch          │
│    - broadMatch          │
│    - narrowMatch         │
└──────────────────────────┘
```

### Module Specifications & Code References
* **Confidence Aggregator**: [`matcher/confidence.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/confidence.py) (`ConfidenceAggregator.aggregate()`)
* **SKOS Mapper**: [`matcher/skos_mapper.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/skos_mapper.py) (`SKOSMapper.determine_relation()`)
* **Output Serializer**: [`matcher/engine.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/matcher/engine.py) (`_export_results()`)

### Code Implementation (`matcher/skos_mapper.py`)
```python
def determine_relation(self, score: float, brsr_concept, gri_concept) -> str:
    if score >= 0.90:
        return "exactMatch"
    elif score >= 0.75:
        return "closeMatch"
    elif score >= 0.55:
        if len(brsr_concept.text) > len(gri_concept.text):
            return "broadMatch"
        return "narrowMatch"
```

### Technical Highlights
1. **W3C SKOS Standard**: Standardizes cross-framework alignments using official SKOS vocabulary (`skos:exactMatch`, `skos:closeMatch`, `skos:broadMatch`, `skos:narrowMatch`).
2. **High-Confidence Filtering**: Confidence threshold ($t \ge 0.35$) yields **79 high-confidence mappings** (53 `broadMatch`, 26 `narrowMatch`).
3. **Mapping Repository Output**: Saved to `data/processed/mapping/mapping_repository.json`.

---

## 5. Validation & Output

```text
┌──────────────────────────┐
│ 5. Validation & Output   │
│  • LLM Verification      │
│  • Explanation           │
│  • Evaluation & Reports  │
└──────────────────────────┘
```

### Module Specifications & Code References
* **LLM Verifier**: [`verifier/llm_verifier.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/verifier/llm_verifier.py) (`LLMVerifier.verify()`)
* **Multi-LLM Evaluator**: [`evaluation/multi_llm_evaluator.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/evaluation/multi_llm_evaluator.py) (`MultiLLMEvaluator.evaluate_mappings()`)
* **Report Manager**: [`evaluation/evaluator.py`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/evaluation/evaluator.py) (`MappingEvaluator.calculate_comparative_metrics()`, `generate_cli_report()`)

### Code Implementation (`verifier/llm_verifier.py` & `evaluator.py`)
```python
# LLM Post-Hoc Verification & Chain-of-Thought (verifier/llm_verifier.py)
prompt = f"Verify candidate alignment between BRSR: '{brsr_label}' and GRI: '{gri_label}'."
response = llm.generate(prompt) # Returns {"verification": "Agree/Disagree", "explanation": "..."}

# Multi-LLM Benchmarking & Pearson Matrix (evaluation/evaluator.py)
df_corr = pd.DataFrame(df_data)
corr_matrix = df_corr.corr(method='pearson')
stats = get_stats(groq_ground_truth, model_predictions)
```

### Technical Highlights
1. **LLM as Auditor, Not Creator**: Restricts LLMs strictly to post-hoc auditing and Chain-of-Thought (CoT) explanation, eliminating hallucinations.
2. **Multi-LLM Benchmarking**: Benchmarks candidate pairs across 9 LLM providers (OpenAI, Groq, Gemini, Mistral, OpenRouter, etc.).
3. **Ground Truth & Metrics**: Uses **Groq (`llama-3.3-70b-versatile`)** as Ground Truth to calculate **Precision**, **Recall**, **F1-Score**, **Accuracy**, and pairwise **Pearson Correlation Matrices** ($r$).
4. **Final Deliverables**: Produces audit reports (`evaluation_report.json`, `evaluation_report.csv`) and terminal summary reports.

---

## 🚀 Commands to Run Pipeline Phases


