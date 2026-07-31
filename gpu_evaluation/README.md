# ESG Ontology Multi-LLM Evaluation Engine (GPU Package)

A portable, self-contained evaluation framework for benchmarking **BRSR–GRI / BRSR–ESRS Semantic Disclosure Alignments** on physical GPU workstations, remote server nodes, and HPC clusters.

---

## 🎯 Overview & Key Features

This package allows anyone pulling the repository onto a GPU machine to execute multi-LLM comparative evaluations using **local open-source GPU models** (via vLLM, Ollama, HuggingFace Transformers, or LM Studio) OR **cloud API providers** (Groq, Google Gemini, OpenAI, DeepSeek, Anthropic Claude).

### Fixed Feature Weight Vector
Disclosure similarity feature vectors $[S_{\text{lexical}}, S_{\text{structural}}, S_{\text{property}}, S_{\text{embedding}}]$ are aggregated using the following fixed weights:

$$\text{Confidence Score} = \left(0.35 \cdot S_{\text{lexical}} + 0.20 \cdot S_{\text{structural}} + 0.15 \cdot S_{\text{property}} + 0.30 \cdot S_{\text{embedding}}\right) \times 100\%$$

- **$w_{\text{lex}}$ (Lexical Overlap):** `0.35`
- **$w_{\text{str}}$ (Structural Hierarchy Path):** `0.20`
- **$w_{\text{prop}}$ (Property & Unit Compatibility):** `0.15`
- **$w_{\text{emb}}$ (Embedding Vector Cosine):** `0.30`

---

## 📁 Repository Structure

```text
gpu_evaluation/
├── configs/                  # Pipeline, open-source model & API key configs
│   ├── settings.yaml         # Master path & dataset parameters
│   ├── providers.yaml        # Local open-source GPU & cloud model endpoints
│   ├── config.yaml           # Execution mode configuration
│   └── .env.example          # Environment variables template
│
├── ontologies/               # RDF Knowledge Graph (.ttl) & schema files
│   ├── brsr/                 # BRSR graph node/edge CSV exports
│   ├── gri/                  # GRI graph node/edge CSV exports
│   ├── merged/               # Final RDF Turtle Ontology (esg_ontology.ttl)
│   └── mappings/             # W3C SKOS Alignment Graphs (mapping.ttl)
│
├── datasets/                 # Mapping datasets & candidate pair feature vectors
│   ├── mapping_repository.json # Evaluated disclosure mapping dataset
│   ├── mapping.json          # SKOS correspondences
│   └── all_candidate_pairs.csv # Candidate similarity feature vectors
│
├── prompts/                  # Prompt templates for verification & model audits
│   ├── verification_prompt.txt # Strict audit prompt for LLM verification layer
│   └── multi_llm_eval_prompt.txt # Inter-model agreement evaluation prompt
│
├── scripts/                  # Core Python modules
│   ├── matcher/              # Matcher engines (lexical, structural, property, reasoning)
│   ├── verifier/             # LLM verification audit engine
│   └── evaluation/           # Metrics calculator & visualizer generator
│
├── utils/                    # Utility scripts & logging framework
│   ├── learner.py            # Feature weight optimization utilities
│   └── logging_config.py     # Logging setup
│
├── outputs/                  # Exported verification reports & metrics JSON/CSV
├── reports/                  # Markdown summary reports & statistics
├── visualizations/           # 6 Publication-quality plot PNGs
├── logs/                     # Detailed execution logs (evaluation.log)
├── run_evaluation.py         # Main CLI execution runner
├── run.sh                    # One-click execution bash script
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment definition
└── README.md                 # Project documentation
```

---

## 💻 System & GPU Hardware Requirements

### Python & CUDA Specification
- **Python Version:** `3.10`, `3.11`, or `3.12`
- **CUDA Version:** `CUDA 11.8` or `CUDA 12.1+`

### Hardware Requirements
- **Local GPU Server:** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g. RTX 3080/4080/4090, A4000/A5000/A100/H100) for hosting open-source models via Ollama or vLLM.
- **Cloud API Execution:** Any CPU machine with internet access for cloud API models.
- **System Memory:** Minimum `16 GB` RAM.

---

## 🛠️ Quick Start on a GPU Workstation

### 1. Clone & Checkout the Branch

```bash
git checkout gpu-evaluation
cd gpu_evaluation
```

### 2. Set Up Virtual Environment

#### Option A: Using Conda
```bash
conda env create -f environment.yml
conda activate esg-gpu-eval
```

#### Option B: Using Pip Virtualenv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Running Evaluation with Open-Source GPU Models

You can run the evaluation using any open-source model running on your GPU server.

### Example 1: Local Ollama GPU Server (`http://localhost:11434`)

```bash
# Run with Llama-3-70B
bash run.sh --provider ollama --model llama3:70b

# Run with Qwen-2.5-72B
bash run.sh --provider ollama --model qwen2.5:72b

# Run with DeepSeek-R1 (14B)
bash run.sh --provider ollama --model deepseek-r1:14b
```

### Example 2: High-Throughput vLLM Server (`http://localhost:8000/v1`)

```bash
bash run.sh --provider vllm --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint http://localhost:8000/v1
```

### Example 3: LM Studio Local Endpoint (`http://localhost:1234/v1`)

```bash
bash run.sh --provider lmstudio --model local-model --endpoint http://localhost:1234/v1
```

---

## ☁️ Running Evaluation with Cloud API Providers

If API keys are configured in `configs/.env`:

```bash
# Groq Llama-3.3-70B
bash run.sh --provider groq --model llama-3.3-70b-versatile

# Google Gemini 2.0 Flash
bash run.sh --provider gemini --model gemini-2.0-flash-exp

# OpenAI GPT-4o-mini
bash run.sh --provider openai --model gpt-4o-mini

# DeepSeek Chat V3
bash run.sh --provider deepseek --model deepseek-chat
```

---

## 📊 Expected Output Artifacts

Upon pipeline execution, output artifacts are saved into relative directories:

| Directory | Output Artifact | Content Description |
|:---|:---|:---|
| **`outputs/`** | `verification_report.json` | Detailed audit decision ("Accepted"/"Rejected"), explanation, and issues per mapping |
| | `evaluation.json` | Precision, Recall, F1-Score, Accuracy, Avg Confidence, Runtime breakdown |
| | `evaluation.csv` | Tabular metrics export |
| **`reports/`** | `summary_report.md` | Human-readable markdown evaluation report |
| | `mapping_statistics.json` | Candidate counts, SKOS relation distribution statistics |
| **`visualizations/`** | `similarity_distribution.png` | Feature vector similarity distributions ($[L, S, P, R]$) |
| | `confidence_distribution.png` | Model confidence score histogram |
| | `mapping_type_distribution.png` | SKOS relation category breakdown |
| | `ontology_coverage.png` | Target disclosure coverage pie chart |
| | `runtime_breakdown.png` | Execution time breakdown per pipeline stage |
| | `confusion_matrix.png` | Matcher prediction vs LLM audit confusion matrix |
| **`logs/`** | `evaluation.log` | Complete timestamped execution log |

---

## 🔧 Troubleshooting Guide

| Issue | Root Cause | Resolution |
|:---|:---|:---|
| `Ollama Connection Refused` | Ollama service is not running | Start Ollama on GPU server: `ollama serve` or `systemctl start ollama`. |
| `vLLM Connection Error` | vLLM server port inactive | Ensure vLLM is launched: `python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --port 8000`. |
| `CUDA Out of Memory` | Model context or batch size exceeds VRAM | Reduce `batch_size` in `configs/settings.yaml` (e.g., set `batch_size: 4`). |
| `API Key Missing` | Environment variable not loaded | Copy `configs/.env.example` to `configs/.env` and insert your API keys. |
