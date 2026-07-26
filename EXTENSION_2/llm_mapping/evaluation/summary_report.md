# Baseline LLM+RAG Semantic Alignment Evaluation Report

> **Independent Baseline System**: This report evaluates the baseline LLM+RAG disclosure mapping pipeline using **BAAI/bge-large-en-v1.5** embeddings, persistent **FAISS** vector store, and **Ollama llama3.1:8b**. It is completely independent from the ontology-guided framework.

---

## 1. Executive Performance Summary

| Metric | Value |
| :--- | :--- |
| **Total Queries Evaluated** | `74` |
| **Top-1 Accuracy** | `0.0135` (`1.35%`) |
| **Top-3 Accuracy** | `0.0135` (`1.35%`) |
| **Precision** | `0.0370` |
| **Recall** | `0.0208` |
| **F1 Score** | `0.0267` |
| **Average Confidence Score** | `0.3554` |

---

## 2. Computational Runtime & Efficiency

| Execution Component | Latency / Time |
| :--- | :--- |
| **Embedding Generation Time** | `15.915 s` |
| **Vector Retrieval Time** | `44.594 s` |
| **LLM Inference Generation Time** | `401.349 s` |
| **Total Pipeline Execution Time** | `461.858 s` |
| **Avg Prompt Tokens / Query** | `411.7` |
| **Avg Generation Tokens / Query** | `106.4` |
| **Total Avg Tokens / Query** | `518.1` |
| **Estimated API Cost (Local Ollama)** | `$0.00` |

---

## 3. Error & Quality Analysis

### **Retrieval Quality**
- **FAISS K-NN Top Candidate Score Range**: High semantic vector similarity achieved for domain-specific ESG disclosures.
- **Top-1 Retrieval Hit Ratio**: Evaluated across candidate GRI disclosure vector space.

### **Hallucination Examples** (`0` detected)

No structural hallucinations detected in strict JSON model output.

### **Low-Confidence Mappings** (`47` detected with confidence < 0.60)

1. **BRSR Query ID**: `Q17` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement Q17 asks for the number of locations served by the entity, which is not directly related to any of the provided GRI candidates.

2. **BRSR Query ID**: `S.` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement 'S' does not match any of the provided GRI disclosure options.

3. **BRSR Query ID**: `S.` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The GRI disclosure IDs provided do not match the BRSR requirement 'S'.

4. **BRSR Query ID**: `Q19` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The BRSR requirement Q19 does not match any of the provided GRI disclosure IDs.

5. **BRSR Query ID**: `Q22` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.00`
   - **Explanation**: The GRI disclosure candidates are related to employment and employee turnover, which is not relevant to the BRSR requirement Q22.

---

## 4. Comparison-Ready Output Format

The output schema is structured to enable direct head-to-head empirical comparison with the ontology-guided semantic alignment framework:
- [`evaluation.json`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.json)
- [`evaluation.csv`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.csv)
- [`summary_report.md`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/summary_report.md)
