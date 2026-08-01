# Automatic Confidence Weight Training Report (EXTENSION_2: BRSR ↔ GRI)

## Executive Summary
This report presents the automatic learning of evidence confidence weights for the **BRSR–GRI Semantic Alignment Engine** in `EXTENSION_2`. Predetermined manual weights (`[0.40, 0.35, 0.15, 0.10]`) were replaced with weights learned via **Logistic Regression** initialized equally at `[1.0, 1.0, 1.0, 1.0]`.

## Data Partitioning & Leakage Prevention
To prevent data leakage, candidate alignment examples were split reproducibly:
- **Training Set (70%):** `511` samples
- **Validation Set (15%):** `109` samples
- **Test Set (15%):** `110` samples (held out completely unseen during training)
- **Random Seed:** `42`

## Initial vs Learned Feature Weights
- **Equal Initialization:** `lexical = 1.0, structural = 1.0, property = 1.0, embedding = 1.0`
- **Learned Normalized Weights ($\sum w_i = 1.0$):**
  - **Lexical ($w_{	ext{lex}}$):** `0.4297` (42.97%)
  - **Structural ($w_{	ext{str}}$):** `0.1935` (19.35%)
  - **Property ($w_{	ext{prop}}$):** `0.0000` (0.00%)
  - **Embedding ($w_{	ext{emb}}$):** `0.3768` (37.68%)

## Feature Importance Ranking Table

| Rank | Feature | Learned Weight | Raw Logistic Coeff |
|:---:|:---|:---:|:---:|
| 1 | Lexical | `0.4297` | `3.8832` |
| 2 | Embedding | `0.3768` | `3.4045` |
| 3 | Structural | `0.1935` | `1.7487` |
| 4 | Property | `0.0000` | `0.0000` |

## Baseline vs Automatically Learned Performance Comparison (Held-Out Test Set)

| Metric | Predetermined Baseline | Automatically Learned | Delta Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | `35.45%` | `42.73%` | `+7.28%` |
| **Precision** | `100.00%` | `100.00%` | `+0.00%` |
| **Recall** | `2.74%` | `13.70%` | `+10.96%` |
| **F1-Score** | `5.33%` | `24.10%` | `+18.77%` |
| **ROC-AUC** | `94.56%` | `96.33%` | `+1.77%` |
