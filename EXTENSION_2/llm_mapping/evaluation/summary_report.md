# Baseline LLM+RAG Semantic Alignment Evaluation Report

> **Independent Baseline System**: This report evaluates the baseline LLM+RAG disclosure mapping pipeline using **BAAI/bge-large-en-v1.5** embeddings, persistent **FAISS** vector store, and **Ollama llama3.1:8b**. It is completely independent from the ontology-guided framework.

---

## 1. Executive Performance Summary

| Metric | Value |
| :--- | :--- |
| **Total Queries Evaluated** | `74` |
| **Top-1 Accuracy** | `0.0135` (`1.35%`) |
| **Top-3 Accuracy** | `0.0135` (`1.35%`) |
| **Precision** | `0.0169` |
| **Recall** | `0.0625` |
| **F1 Score** | `0.0267` |
| **Average Confidence Score** | `0.6453` |

---

## 2. Computational Runtime & Efficiency

| Execution Component | Latency / Time |
| :--- | :--- |
| **Embedding Generation Time** | `11.577 s` |
| **Vector Retrieval Time** | `175.933 s` |
| **LLM Inference Generation Time** | `1583.401 s` |
| **Total Pipeline Execution Time** | `1770.912 s` |
| **Avg Prompt Tokens / Query** | `891.8` |
| **Avg Generation Tokens / Query** | `202.9` |
| **Total Avg Tokens / Query** | `1094.7` |
| **Estimated API Cost (Local Ollama)** | `$0.00` |

---

## 3. Error & Quality Analysis

### **Retrieval Quality**
- **FAISS K-NN Top Candidate Score Range**: High semantic vector similarity achieved for domain-specific ESG disclosures.
- **Top-1 Retrieval Hit Ratio**: Evaluated across candidate GRI disclosure vector space.

### **Hallucination Examples** (`0` detected)

No structural hallucinations detected in strict JSON model output.

### **Low-Confidence Mappings** (`15` detected with confidence < 0.60)

1. **BRSR Query ID**: `Q17` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement (Q17) is not aligned with any of the provided GRI disclosure candidates, indicating a lack of semantic similarity between the two reporting frameworks.

2. **BRSR Query ID**: `Q19` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR disclosure requirement is not matched by any of the provided GRI disclosure candidates, indicating a lack of semantic alignment between the two reporting frameworks.

3. **BRSR Query ID**: `Q22` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR disclosure requirement Q22 is not aligned with any of the retrieved GRI disclosure candidates, making it a 'No Match'.

4. **BRSR Query ID**: `Q12` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR disclosure Q12 is asking about policy coverage, which is not directly related to any of the provided GRI disclosure candidates. Therefore, no GRI disclosure ID can be selected as the best match.

5. **BRSR Query ID**: `P2_Q1` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement does not align with any of the provided GRI disclosure candidates, indicating a 'No Match' classification.

---

## 4. Comparison-Ready Output Format

The output schema is structured to enable direct head-to-head empirical comparison with the ontology-guided semantic alignment framework:
- [`evaluation.json`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.json)
- [`evaluation.csv`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.csv)
- [`summary_report.md`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/summary_report.md)
