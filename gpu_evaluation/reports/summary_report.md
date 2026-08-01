# GPU Evaluation Verification Report (`EXTENSION_2` Reference)

## Hardware & Model Runtime Metadata
- **Provider:** `vllm`
- **Model:** `meta-llama/Meta-Llama-3-70B-Instruct`
- **Endpoint:** `http://localhost:8000/v1`
- **Execution Time:** `0.01s` (LLM Inference: `0.00s`)

## Feature Weighting Configuration (Automatically Learned)
- **Lexical Weight ($w_{	ext{lex}}$):** `0.4297` (42.97%)
- **Structural Weight ($w_{	ext{str}}$):** `0.1935` (19.35%)
- **Property Weight ($w_{	ext{prop}}$):** `0.0000` (0.00%)
- **Embedding Weight ($w_{	ext{emb}}$):** `0.3768` (37.68%)

## Verification Audit Summary
- **Total Disclosure Mappings Evaluated:** `79`
- **Accepted Alignments (Confidence $\ge 25\%$):** `66` (83.5%)
- **Rejected Alignments:** `13`
- **Average Learned Confidence Score:** `27.96%`

## Feature Importance Ranking Table

| Rank | Feature | Learned Weight | Target Contribution |
|:---:|:---|:---:|:---:|
| 1 | Lexical | `0.4297` | **42.97%** |
| 2 | Embedding | `0.3768` | **37.68%** |
| 3 | Structural | `0.1935` | **19.35%** |
| 4 | Property | `0.0000` | **0.00%** |

---
*Report generated automatically by `gpu_evaluation/run_evaluation.py` on 2026-08-01 23:08:03*
