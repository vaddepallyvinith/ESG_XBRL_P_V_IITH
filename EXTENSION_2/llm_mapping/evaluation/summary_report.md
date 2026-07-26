# Baseline LLM+RAG Semantic Alignment Evaluation Report

> **Independent Baseline System**: This report evaluates the baseline LLM+RAG disclosure mapping pipeline using **BAAI/bge-large-en-v1.5** embeddings, persistent **FAISS** vector store, and **Ollama llama3.1:8b**. It is completely independent from the ontology-guided framework.

---

## 1. Executive Performance Summary

| Metric | Value |
| :--- | :--- |
| **Total Queries Evaluated** | `74` |
| **Top-1 Accuracy** | `0.0270` (`2.70%`) |
| **Top-3 Accuracy** | `0.0270` (`2.70%`) |
| **Precision** | `0.0290` |
| **Recall** | `0.2857` |
| **F1 Score** | `0.0526` |
| **Average Confidence Score** | `0.6839` |

---

## 2. Computational Runtime & Efficiency

| Execution Component | Latency / Time |
| :--- | :--- |
| **Embedding Generation Time** | `12.356 s` |
| **Vector Retrieval Time** | `52.835 s` |
| **LLM Inference Generation Time** | `475.518 s` |
| **Total Pipeline Execution Time** | `540.710 s` |
| **Avg Prompt Tokens / Query** | `326.4` |
| **Avg Generation Tokens / Query** | `98.6` |
| **Total Avg Tokens / Query** | `425.0` |
| **Estimated API Cost (Local Ollama)** | `$0.00` |

---

## 3. Error & Quality Analysis

### **Retrieval Quality**
- **FAISS K-NN Top Candidate Score Range**: High semantic vector similarity achieved for domain-specific ESG disclosures.
- **Top-1 Retrieval Hit Ratio**: Evaluated across candidate GRI disclosure vector space.

### **Hallucination Examples** (`1` detected)

1. **BRSR Query ID**: `P6_Q3`
   - **Hallucinated Output**: `Topic 11.22 Public policy`
   - **Type**: `Close Match`
   - **Explanation**: Alignment derived from top RAG candidate.

### **Low-Confidence Mappings** (`4` detected with confidence < 0.60)

1. **BRSR Query ID**: `S.` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.15`
   - **Explanation**: The BRSR 'General Disclosures' requirement is too generic to match any specific GRI topic disclosure, resulting in a No Match with low confidence.

2. **BRSR Query ID**: `S.` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.10`
   - **Explanation**: The generic nature of BRSR 'S.' with no defined requirements or metrics prevents alignment with any GRI disclosure. GRI standards demand specific reporting elements (e.g., privacy policies, non-compliance counts) absent here.

3. **BRSR Query ID**: `Q19` -> **GRI**: `Disclosure 2-9 Governance structure and composition` (`Narrow Match`)
   - **Confidence**: `0.45`
   - **Explanation**: The BRSR Q19 on women's participation is a specific diversity metric that could fall under the broader governance structure disclosure (GRI 2-9). However, the GRI requirement lacks explicit gender-related content, making this a 'Narrow Match' where the BRSR requirement is a subset of the GRI's governance scope. The low confidence reflects the absence of direct gender-related language in the GRI candidate.

4. **BRSR Query ID**: `Q12` -> **GRI**: `None` (`No Match`)
   - **Confidence**: `0.15`
   - **Explanation**: BRSR Q12 mandates disclosure based on policy coverage gaps, but no GRI candidate addresses this specific condition. The closest candidates discuss mechanisms or governance, which are unrelated to the policy coverage trigger.

---

## 4. Comparison-Ready Output Format

The output schema is structured to enable direct head-to-head empirical comparison with the ontology-guided semantic alignment framework:
- [`evaluation.json`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.json)
- [`evaluation.csv`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/evaluation.csv)
- [`summary_report.md`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/EXTENSION_2/llm_mapping/evaluation/summary_report.md)
