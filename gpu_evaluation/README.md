# ESG Ontology Multi-LLM Evaluation Engine (GPU Package)

A fully self-contained, portable benchmark evaluation package for the **BRSR–GRI / BRSR–ESRS Semantic Alignment Pipeline** designed to execute seamlessly on physical GPU workstations and cloud compute instances.

---

## 📋 Project Overview

This framework evaluates semantic disclosure mappings generated between ESG regulatory standards (SEBI BRSR, GRI Standards, and European ESRS) using an **ESG Resource Ontology (RSO)** knowledge graph and **Automatically Learned Confidence Weights**. 

The pipeline performs multi-LLM comparative audits across multiple providers (Groq, Google Gemini, OpenAI GPT-4o, Ollama, DeepSeek, Claude, OpenRouter, Cerebras) to benchmark alignment accuracy, inter-model agreement, reasoning validity, token usage, and cost efficiency.

---

## 📁 Directory Structure

```text
gpu_evaluation/
├── configs/                  # Pipeline, provider, and API key configurations
│   ├── settings.yaml         # Master dataset, path, and execution parameters
│   ├── providers.yaml        # LLM provider settings (Groq, Gemini, OpenAI, Ollama, etc.)
│   ├── config.yaml           # Pipeline execution mode settings
│   └── .env.example          # Environment variables template for API keys
│
├── ontologies/               # RDF Knowledge Graph (.ttl) and ontology schemas
│   ├── brsr/                 # BRSR node and relationship CSV graph exports
│   ├── gri/                  # GRI node and relationship CSV graph exports
│   ├── merged/               # Final merged RDF Turtle Ontology (esg_ontology.ttl)
│   └── mappings/             # Standard W3C SKOS alignment turtle graphs (mapping.ttl)
│
├── datasets/                 # Mapping datasets and learned feature weights
│   ├── mapping_repository.json # Complete dataset of evaluated disclosure mappings
│   ├── mapping.json          # SKOS candidate pair correspondences
│   ├── learned_weights.json  # Learned feature weight vector [L, S, P, R]
│   └── candidate_pairs.csv   # Raw candidate pair feature vectors
│
├── prompts/                  # Prompt templates for verification & multi-LLM evaluation
│   ├── verification_prompt.txt # Strict audit prompt for LLM verification layer
│   └── multi_llm_eval_prompt.txt # Benchmark agreement prompt across models
│
├── scripts/                  # Python source modules
│   ├── matcher/              # Lexical, structural, property & reasoning matchers
│   ├── verifier/             # LLM verification audit engine
│   └── evaluation/           # Evaluation metrics calculator & visualization generator
│
├── utils/                    # Utility scripts, logging & confidence weight learner
│   ├── learner.py            # Logistic regression & grid search weight optimization
│   └── logging_config.py     # Centralized logger utility
│
├── cache/                    # Local embeddings and response cache
├── checkpoints/              # Multi-provider benchmark evaluation checkpoints
├── outputs/                  # JSON & CSV output evaluation reports
│   ├── verification_report.json
│   ├── evaluation.json
│   └── evaluation.csv
│
├── reports/                  # Markdown & summary statistic reports
│   ├── summary_report.md
│   └── mapping_statistics.json
│
├── visualizations/           # 6 Publication-quality evaluation plots (.png)
│   ├── similarity_distribution.png
│   ├── confidence_distribution.png
│   ├── mapping_type_distribution.png
│   ├── ontology_coverage.png
│   ├── runtime_breakdown.png
│   └── confusion_matrix.png
│
├── logs/                     # Detailed execution logs (evaluation.log)
├── run_evaluation.py         # Main CLI evaluation runner script
├── run.sh                    # One-click execution bash script
├── requirements.txt          # Python pip dependencies
├── environment.yml           # Conda environment definition
└── README.md                 # Project documentation
```

---

## 💻 Hardware & System Requirements

### Python & CUDA Specification
- **Python Version:** `3.10`, `3.11`, or `3.12`
- **CUDA Driver Version:** `CUDA 11.8` or `CUDA 12.1+`

### System Hardware
- **GPU Requirements:** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g., RTX 3080/4080/4090, A4000/A5000/A100/H100) if running local LLMs via Ollama (`llama3:70b` / `mistral`).
- **Cloud API Execution:** If using API providers (Groq, Gemini, OpenAI), CPU execution is supported.
- **RAM Requirements:** Minimum `16 GB` System RAM.
- **Disk Space:** Minimum `5 GB` free space.

---

## 🛠️ Installation & Setup

### Option 1: Conda Environment Setup (Recommended)

```bash
# Clone the repository and switch to the gpu-evaluation branch
git checkout gpu-evaluation

# Navigate to the evaluation folder
cd gpu_evaluation

# Create and activate conda environment
conda env create -f environment.yml
conda activate esg-gpu-eval
```

### Option 2: Pip Virtual Environment Setup

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration & API Keys

1. Copy the environment variables template:
   ```bash
   cp configs/.env.example configs/.env
   ```

2. Edit `configs/.env` to include your provider API keys (optional if running locally via Ollama):
   ```env
   GROQ_API_KEY=gsk_your_groq_key_here
   GEMINI_API_KEY=AIzaSy_your_gemini_key_here
   OPENAI_API_KEY=sk-proj-your_openai_key_here
   DEEPSEEK_API_KEY=sk-your_deepseek_key_here
   ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
   OLLAMA_HOST=http://localhost:11434
   ```

---

## 🚀 Execution Instructions

Run the one-click execution script:

```bash
bash run.sh
```

Or run directly using Python:

```bash
python3 run_evaluation.py --config configs/settings.yaml
```

---

## 📊 Expected Outputs

Upon completion, all evaluation results will be automatically exported to relative paths:

| Directory | Generated Artifacts | Description |
|:---|:---|:---|
| **`outputs/`** | `verification_report.json` | Detailed per-mapping audit decisions, explanations & issues |
| | `evaluation.json` | Global Precision, Recall, F1, Accuracy & Runtime Breakdown |
| | `evaluation.csv` | Tabular metrics export |
| **`reports/`** | `summary_report.md` | Human-readable markdown evaluation report |
| | `mapping_statistics.json` | Candidate counts, SKOS relation distribution statistics |
| **`visualizations/`** | `similarity_distribution.png` | Feature vector similarity distributions ($[L, S, P, R]$) |
| | `confidence_distribution.png` | Model confidence score histogram |
| | `mapping_type_distribution.png` | SKOS relation category breakdown |
| | `ontology_coverage.png` | Target disclosure coverage pie chart |
| | `runtime_breakdown.png` | Execution time breakdown per pipeline stage |
| | `confusion_matrix.png` | Prediction vs LLM Verification Confusion Matrix |
| **`logs/`** | `evaluation.log` | Timestamped execution logs |

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Solution |
|:---|:---|:---|
| `ModuleNotFoundError: No module named 'rdflib'` | Missing Python package | Run `pip install -r requirements.txt` inside your virtual environment. |
| `APIKeyError` or `AuthenticationFailed` | Invalid or missing API key | Ensure your API key is correctly specified in `configs/.env`. |
| `Ollama Connection Refused` | Local Ollama service is down | Start local Ollama service: `ollama serve` or switch default provider to `groq` in `configs/settings.yaml`. |
| `CUDA Out of Memory` | Batch size too large for GPU VRAM | Reduce `batch_size` in `configs/settings.yaml` (e.g. set `batch_size: 4`). |
