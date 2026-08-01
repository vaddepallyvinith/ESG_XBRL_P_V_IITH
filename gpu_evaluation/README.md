# ESG Ontology Multi-LLM Evaluation Engine (EXTENSION_2: BRSR ↔ GRI)

A portable, self-contained evaluation framework for benchmarking **BRSR–GRI Semantic Disclosure Alignments** on physical GPU workstations, remote server nodes, and HPC clusters using **Automatically Learned Feature Weights**.

---

## 🎯 Overview & Key Features

This package allows anyone pulling the repository onto a GPU machine to execute multi-LLM comparative evaluations for **EXTENSION_2 (BRSR ↔ GRI Standards)** using **local open-source GPU models** (via vLLM, Ollama, HuggingFace Transformers, or LM Studio) OR **cloud API providers** (Groq, Google Gemini, OpenAI, DeepSeek, Anthropic Claude).

### Automatically Learned Feature Weight Vector (`EXTENSION_2`)
Disclosure similarity feature vectors $[S_{\text{lexical}}, S_{\text{structural}}, S_{\text{property}}, S_{\text{embedding}}]$ are aggregated using the feature weights automatically learned via **Logistic Regression** trained over ground-truth alignment data (initialized equally at $[1.0, 1.0, 1.0, 1.0]$ with a 70/15/15 train/val/test split to prevent data leakage):

$$\text{Confidence Score} = \left(0.4297 \cdot S_{\text{lexical}} + 0.1935 \cdot S_{\text{structural}} + 0.0000 \cdot S_{\text{property}} + 0.3768 \cdot S_{\text{embedding}}\right) \times 100\%$$

- **$w_{\text{lex}}$ (Lexical Similarity):** `0.4297` ($42.97\%$)
- **$w_{\text{str}}$ (Structural Hierarchy Path):** `0.1935` ($19.35\%$)
- **$w_{\text{prop}}$ (Property & Unit Compatibility):** `0.0000` ($0.00\%$)
- **$w_{\text{emb}}$ (Embedding Vector Cosine):** `0.3768` ($37.68\%$)

---

## 📁 Repository Structure

```text
gpu_evaluation/
├── configs/                  # Pipeline, open-source model & API key configs
│   ├── settings.yaml         # Master path & learned weights parameters
│   ├── providers.yaml        # Local open-source GPU & cloud model endpoints
│   ├── learned_weights.json  # Learned weights JSON artifact
│   ├── config.yaml           # Execution mode configuration
│   └── .env.example          # Environment variables template
│
├── ontologies/               # RDF Knowledge Graph (.ttl) & schema files (EXTENSION_2)
│   ├── brsr/                 # BRSR graph node/edge CSV exports
│   ├── gri/                  # GRI graph node/edge CSV exports
│   ├── merged/               # Final RDF Turtle Ontology (esg_ontology.ttl - 5.67 MB)
│   └── mappings/             # W3C SKOS Alignment Graphs (mapping.ttl)
│
├── datasets/                 # EXTENSION_2 BRSR ↔ GRI Mapping Datasets & Training Reports
│   ├── learned_weights.json  # Learned feature weight vector configuration
│   ├── confidence_training_report.json # Weight training metrics report
│   ├── confidence_training_report.md   # Markdown weight training report
│   ├── feature_importance.csv          # Feature importance ranking table
│   ├── baseline_vs_learned.csv         # Baseline vs learned performance gains table
│   ├── mapping_repository.json # Evaluated 79 BRSR-GRI disclosure mappings
│   ├── brsr_gri_mapping_repository.json # Secondary mapping repository
│   ├── mapping_summary.csv   # Tabular summary of BRSR-GRI correspondences
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

#### Option B: Using Pip & Virtualenv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Running Evaluation on Open-Source GPU Models

### 1. Ollama (Local Open-Source GPU Inference)
If your GPU server has [Ollama](https://ollama.ai) installed:

```bash
# Pull model
ollama pull llama3:70b

# Run evaluation using learned weights
bash run.sh --provider ollama --model llama3:70b
```

### 2. vLLM Server (High-Throughput Open-Source GPU Serving)
If running a [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible server:

```bash
# Launch vLLM server on GPU
vllm serve meta-llama/Meta-Llama-3-70B-Instruct --port 8000

# Execute evaluation script
bash run.sh --provider vllm --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint http://localhost:8000/v1
```

### 3. LM Studio / LocalAI Endpoint
```bash
bash run.sh --provider lmstudio --model local-model --endpoint http://localhost:1234/v1
```

### 4. Cloud API Providers (Groq, Gemini, OpenAI, DeepSeek)
```bash
# Set your API keys in environment or .env
export GROQ_API_KEY="your-groq-key"

# Run Groq Llama-3.3-70B evaluation
bash run.sh --provider groq --model llama-3.3-70b-versatile
```

---

## 📊 Baseline vs Automatically Learned Performance Gain (`EXTENSION_2`)

| Performance Metric | Predetermined Baseline (`[0.40, 0.35, 0.15, 0.10]`) | Automatically Learned (`[0.43, 0.19, 0.00, 0.38]`) | Delta Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | `35.45%` | **`42.73%`** | **`+7.28%`** |
| **Precision** | `100.00%` | **`100.00%`** | **`+0.00%`** |
| **Recall** | `2.74%` | **`13.70%`** | **`+10.96%`** |
| **F1-Score** | `5.33%` | **`24.10%`** | **`+18.77%`** |
| **ROC-AUC** | `94.56%` | **`96.33%`** | **`+1.77%`** |

---

## 📄 Output Artifacts & Reports

After running the evaluation, outputs are generated in:
- `outputs/verification_report.json` — Detailed per-disclosure LLM verification audit results
- `outputs/verification_report.csv` — Tabular CSV verification audit format
- `reports/summary_report.md` — Complete markdown summary with execution timing and feature weights
- `logs/evaluation.log` — Execution log file
