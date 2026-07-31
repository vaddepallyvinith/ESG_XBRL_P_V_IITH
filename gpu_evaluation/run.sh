#!/usr/bin/env bash
# ==============================================================================
# GPU Workstation Multi-LLM Evaluation Execution Script
# Fully self-contained, automated file validation, evaluation, and report export.
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_BIN=$(which python3)
if [ -f "$SCRIPT_DIR/../venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../venv/bin/python"
elif command -v python &> /dev/null; then
    PYTHON_BIN=$(which python)
fi

echo "======================================================================"
echo "    ESG Ontology Multi-LLM Evaluation Framework (GPU Workstation)    "
echo "======================================================================"
echo "Using Python: $PYTHON_BIN"

# 1. Create Required Output & Log Directories
echo "📁 Step 1: Validating Directory Structure..."
mkdir -p configs ontologies/merged ontologies/brsr ontologies/gri ontologies/mappings datasets prompts scripts utils cache checkpoints outputs reports visualizations logs

# 2. Check Python Environment & Dependencies
echo "🐍 Step 2: Checking Python Dependencies..."
if ! "$PYTHON_BIN" -c "import rdflib, torch, pandas, yaml, sklearn" 2>/dev/null; then
    echo "⚠️ Installing requirements..."
    "$PYTHON_BIN" -m pip install -r requirements.txt --quiet || true
else
    echo "✅ All required Python packages are installed."
fi

# 3. Validate Ontology and Dataset Files
echo "🔍 Step 3: Validating Required Data & Ontology Files..."
if [ ! -f "ontologies/merged/esg_ontology.ttl" ]; then
    echo "⚠️ Warning: esg_ontology.ttl not found in ontologies/merged/."
else
    echo "✅ Found Ontology Graph: ontologies/merged/esg_ontology.ttl"
fi

if [ ! -f "datasets/mapping_repository.json" ] && [ ! -f "datasets/mapping.json" ]; then
    echo "❌ Error: Mapping dataset missing in datasets/ folder."
    exit 1
else
    echo "✅ Found Mapping Repository Dataset."
fi

if [ ! -f "configs/.env" ]; then
    if [ -f "configs/.env.example" ]; then
        echo "ℹ️ Copying configs/.env.example to configs/.env..."
        cp configs/.env.example configs/.env
    fi
fi

# 4. Execute Multi-LLM Evaluation Pipeline
echo "🚀 Step 4: Starting Multi-LLM Evaluation Pipeline..."
"$PYTHON_BIN" run_evaluation.py --config configs/settings.yaml

# 5. Display Summary Report
echo "======================================================================"
echo "✅ Evaluation Execution Complete!"
echo "Outputs directory:      $SCRIPT_DIR/outputs/"
echo "Reports directory:      $SCRIPT_DIR/reports/"
echo "Visualizations:         $SCRIPT_DIR/visualizations/"
echo "Logs:                   $SCRIPT_DIR/logs/evaluation.log"
echo "======================================================================"
