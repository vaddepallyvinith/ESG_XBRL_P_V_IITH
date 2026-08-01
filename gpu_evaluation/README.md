# ESG Ontology Multi-LLM Evaluation Engine (`EXTENSION_2`: BRSR ↔ GRI)

A self-contained, portable evaluation framework for benchmarking **BRSR–GRI Semantic Disclosure Alignments** on physical GPU workstations, server nodes, and HPC clusters using **Automatically Learned Feature Weights**.

---

## 🎯 Overview & Feature Weighting Model

This package allows anyone pulling the repository onto a GPU machine to execute multi-LLM comparative evaluations for **`EXTENSION_2` (BRSR ↔ GRI Standards)** using **local open-source GPU models** (vLLM, Ollama, HuggingFace, LM Studio) OR **cloud API providers** (Groq, Google Gemini, OpenAI, DeepSeek, Anthropic Claude).

### Automatically Learned Feature Weight Vector
Disclosure similarity feature vectors $[S_{\text{lexical}}, S_{\text{structural}}, S_{\text{property}}, S_{\text{embedding}}]$ are aggregated using feature weights automatically learned via **Logistic Regression** trained over ground-truth alignment data (initialized equally at $[1.0, 1.0, 1.0, 1.0]$ with a 70/15/15 train/val/test split to prevent data leakage):

$$\text{Confidence Score} = \left(0.4297 \cdot S_{\text{lexical}} + 0.1935 \cdot S_{\text{structural}} + 0.0000 \cdot S_{\text{property}} + 0.3768 \cdot S_{\text{embedding}}\right) \times 100\%$$

- **$w_{\text{lex}}$ (Lexical Overlap):** `0.4297` ($42.97\%$)
- **$w_{\text{str}}$ (Structural Hierarchy):** `0.1935` ($19.35\%$)
- **$w_{\text{prop}}$ (Property & Unit Compatibility):** `0.0000` ($0.00\%$)
- **$w_{\text{emb}}$ (Embedding Vector Cosine):** `0.3768` ($37.68\%$)

---

## 📖 Step-by-Step Guide to Run on a GPU Machine

Follow these step-by-step instructions to set up and run the evaluation pipeline on your GPU workstation.

### Step 1: Clone Repository & Checkout Branch

Open your GPU machine terminal and execute:

```bash
git clone https://github.com/vaddepallyvinith/ESG_XBRL_P_V_IITH.git
cd ESG_XBRL_P_V_IITH
git checkout gpu-evaluation
cd gpu_evaluation
```

---

### Step 2: Virtual Environment & Dependency Setup

Choose either **Conda** or **Pip Virtualenv** to set up dependencies.

#### Method A: Using Conda (Recommended for CUDA GPU Servers)
```bash
conda env create -f environment.yml
conda activate esg-gpu-eval
```

#### Method B: Using Pip Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Choose Your Model Provider & Start Inference Server

Select one of the four execution modes depending on your GPU machine environment:

#### Mode 1: Ollama Local GPU Inference (Simplest for Workstations)
If your GPU machine has [Ollama](https://ollama.ai) installed:

```bash
# 1. Pull desired open-source model into VRAM
ollama pull llama3:70b   # or ollama pull qwen2.5:72b / deepseek-r1:14b

# 2. Run evaluation script
bash run.sh --provider ollama --model llama3:70b
```

#### Mode 2: vLLM High-Throughput GPU Serving (Fastest for Multi-GPU Nodes)
If serving models via [vLLM](https://github.com/vllm-project/vllm):

```bash
# 1. Launch vLLM server on GPU
vllm serve meta-llama/Meta-Llama-3-70B-Instruct --port 8000 --tensor-parallel-size 2

# 2. Run evaluation pointing to local vLLM endpoint
bash run.sh --provider vllm --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint http://localhost:8000/v1
```

#### Mode 3: LM Studio / LocalAI OpenAI-Compatible GUI
If using LM Studio on a local GPU workstation:

```bash
# 1. Start LM Studio Local Server (default port 1234)
# 2. Execute evaluation
bash run.sh --provider lmstudio --model local-model --endpoint http://localhost:1234/v1
```

#### Mode 4: Cloud API Providers (Groq / Gemini / OpenAI / DeepSeek)
If testing against cloud LLM APIs:

```bash
# 1. Export API key
export GROQ_API_KEY="your-groq-api-key"
# or export GEMINI_API_KEY="your-gemini-key" / OPENAI_API_KEY="your-openai-key"

# 2. Execute evaluation
bash run.sh --provider groq --model llama-3.3-70b-versatile
```

---

### Step 4: Run Direct Python CLI (Alternative to `run.sh`)

You can also run `run_evaluation.py` directly with custom command-line options:

```bash
python run_evaluation.py \
  --provider ollama \
  --model llama3:70b \
  --endpoint http://localhost:11434 \
  --config configs/settings.yaml
```

#### Supported CLI Parameters:
- `--provider`: `ollama` | `vllm` | `lmstudio` | `groq` | `gemini` | `openai` | `deepseek`
- `--model`: Open-source or cloud model identifier (e.g. `llama3:70b`, `qwen2.5:72b`, `llama-3.3-70b-versatile`)
- `--endpoint`: API base URL endpoint (e.g. `http://localhost:11434`, `http://localhost:8000/v1`)
- `--config`: Path to settings configuration file (default: `configs/settings.yaml`)

---

### Step 5: Verify Generated Output Reports & Artifacts

Upon completion of the evaluation script, review the generated outputs in the following paths:

| Output Artifact Path | Description |
|:---|:---|
| [`outputs/verification_report.json`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/gpu_evaluation/outputs/verification_report.json) | Full per-disclosure LLM verification audit decisions and reasoning |
| [`outputs/verification_report.csv`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/gpu_evaluation/outputs/verification_report.csv) | Tabular CSV export of alignment verification scores |
| [`reports/summary_report.md`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/gpu_evaluation/reports/summary_report.md) | Summary markdown report detailing execution timing & learned weights |
| [`datasets/learned_weights.json`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/gpu_evaluation/datasets/learned_weights.json) | Learned weight vector JSON artifact |
| [`datasets/baseline_vs_learned.csv`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/gpu_evaluation/datasets/baseline_vs_learned.csv) | Baseline vs learned accuracy comparison table |
| [`logs/evaluation.log`](file:///home/vinith/A/INTENSHIPS/IIT-H/Nested_RAG/gpu_evaluation/logs/evaluation.log) | Complete execution log file |

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

## 📁 Complete Repository Hierarchy

```text
gpu_evaluation/
├── configs/                  # Settings, model endpoints & learned weights configs
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
├── visualizations/           # Publication-quality plot PNGs
├── logs/                     # Detailed execution logs (evaluation.log)
├── run_evaluation.py         # Main CLI execution runner
├── run.sh                    # One-click execution bash script
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment definition
└── README.md                 # Project documentation
```
