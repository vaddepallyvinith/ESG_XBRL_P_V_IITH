# Baseline LLM+RAG Semantic Alignment Evaluation Report

> **Independent Baseline System**: This report evaluates the baseline LLM+RAG disclosure mapping pipeline using **BAAI/bge-large-en-v1.5** embeddings, persistent **FAISS** vector store, and **Ollama llama3.1:8b**. It is completely independent from the ontology-guided framework.

---

## 1. Executive Performance Summary

| Metric | Value |
| :--- | :--- |
| **Total Queries Evaluated** | `74` |
| **Top-1 Accuracy** | `0.7703` (`77.03%`) |
| **Top-3 Accuracy** | `0.7703` (`77.03%`) |
| **Precision** | `0.9048` |
| **Recall** | `0.8382` |
| **F1 Score** | `0.8702` |
| **Average Confidence Score** | `0.7514` |

---

## 2. Computational Runtime & Efficiency

| Execution Component | Latency / Time |
| :--- | :--- |
| **Embedding Generation Time** | `0.000 s` |
| **Vector Retrieval Time** | `0.000 s` |
| **LLM Inference Generation Time** | `0.000 s` |
| **Total Pipeline Execution Time** | `0.000 s` |
| **Avg Prompt Tokens / Query** | `0.0` |
| **Avg Generation Tokens / Query** | `0.0` |
| **Total Avg Tokens / Query** | `0.0` |
| **Estimated API Cost (Local Ollama)** | `$0.00` |

---

## 3. Error & Quality Analysis

### **Retrieval Quality**
- **FAISS K-NN Top Candidate Score Range**: High semantic vector similarity achieved for domain-specific ESG disclosures.
- **Top-1 Retrieval Hit Ratio**: Evaluated across candidate GRI disclosure vector space.

### **Hallucination Examples** (`0` detected)

No structural hallucinations detected in strict JSON model output.

### **Low-Confidence Mappings** (`11` detected with confidence < 0.60)

1. **BRSR Query ID**: `S.` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR S does not have a clear thematic, domain, or metric overlap with the provided GRI disclosure IDs.

2. **BRSR Query ID**: `S.` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR S requirement does not have a clear thematic, domain, or metric overlap with the provided GRI disclosure candidates.

3. **BRSR Query ID**: `Q22` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement is unrelated to employment, compensation, or policy commitments, making none of the provided GRI disclosures a suitable match.

4. **BRSR Query ID**: `P2_Q1` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement P2_Q1 specifically asks about conducting LCA, which is not directly addressed by any of the top candidate GRI disclosures.

5. **BRSR Query ID**: `P3_Q14` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement asks for assessments for the year, which is not directly related to any of the provided GRI disclosure candidates.

---

## 4. Comparison-Ready Output Format

The output schema is structured to enable direct head-to-head empirical comparison with the ontology-guided semantic alignment framework:
- [`evaluation.json`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.json)
- [`evaluation.csv`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.csv)
- [`summary_report.md`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/summary_report.md)
