# Real-Time Crisis Detection System

> End-to-end NLP pipeline that monitors social media and news feeds, classifies crisis events into **7 categories** using a fine-tuned **XLM-RoBERTa** model, and visualizes detected crises on an interactive map.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFAA00?style=flat-square&logo=huggingface&logoColor=black)
![Accuracy](https://img.shields.io/badge/Val_Accuracy-96.7%25-10B981?style=flat-square)
![F1 Score](https://img.shields.io/badge/Macro_F1-0.96-3B82F6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-F59E0B?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Colab_T4-9333EA?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

This system provides **real-time detection and classification of crisis events** (floods, earthquakes, fires, accidents, violence, storms) from multilingual social media and news sources. Designed for emergency response coordinators, it aggregates raw posts, classifies crisis type with high confidence, clusters related events geographically, and displays them on an interactive map.

| Detail | Value |
|--------|-------|
| **Target Region** | Indonesia (global capability via multilingual model) |
| **Data Sources** | Reddit, Telegram, BMKG (Indonesia Met Agency), RSS News Feeds |
| **Crisis Categories** | flood, earthquake, fire, accident, violence, storm, other |
| **Classification Coverage** | 96% at confidence ≥ 0.7 |

---

## Key Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.7%** |
| **Test Set Accuracy** | **96.0%** |
| **Macro F1 Score** | **0.96** |
| **Training Samples** | 28,415 |
| **Model Parameters** | 278M |

### Per-Class Performance

| Crisis Type | Precision | Recall | F1-Score |
|:------------|:---------:|:------:|:--------:|
| flood | 0.95 | 0.97 | 0.96 |
| earthquake | 0.99 | 1.00 | **1.00** 🎉 |
| fire | 1.00 | 1.00 | **1.00** 🎉 |
| accident | 0.96 | 0.97 | 0.96 |
| violence | 0.95 | 0.96 | 0.96 |
| storm | 0.94 | 0.97 | 0.95 |
| other | 0.94 | 0.86 | 0.90 |

### SHAP Explainability Examples

The model provides **word-level attribution** explaining each prediction:

| Input Post | Prediction | Top SHAP Features |
|------------|-----------|-------------------|
| *"roads completely flooded in jakarta selatan"* | **flood** (99.9%) | `flood` +0.533, `road` +0.464 |
| *"Earthquake M5.2 at depth 10km in PALU-SULTENG"* | **earthquake** (99.9%) | `km` +0.353, `Tenggara` +0.146 |

---

## System Architecture

```
+------------------------------------------------------------------+
|                    DATA COLLECTION LAYER                         |
|  Reddit API · BMKG RSS · Telegram · News RSS · Bulk Datasets     |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|                    PREPROCESSING (NB02)                          |
|  Text cleaning · Geo-resolution · Timestamp normalization        |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|               CLASSIFICATION MODEL (NB04)  ← CORE               |
|  XLM-RoBERTa-base fine-tuned · 7-class · 96.7% accuracy         |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|          CREDIBILITY & CLUSTERING (NB05)                         |
|  SBERT embeddings · DBSCAN clustering · Credibility scoring      |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|         EXPLAINABILITY & EVALUATION (NB06)                       |
|  SHAP word-level attribution · Classification report             |
+------------------------------------------------------------------+
                               |
+------------------------------------------------------------------+
|              VISUALIZATION DASHBOARD (NB07)                      |
|  Folium interactive map · GeoJSON export · Event timeline        |
+------------------------------------------------------------------+
```

---

## Project Structure

```
crisis-detection-system/
├── notebooks/
│   ├── 01_data_collection.ipynb              # Live data from Reddit/BMKG/RSS
│   ├── 01b_bulk_dataset_loader.ipynb         # 28K+ labeled training data
│   ├── 01b_earthquake_enrichment.py          # Earthquake data boost (69 → 3,966)
│   ├── 01b_fire_enrichment.py                # Fire data boost (379 → 5,000)
│   ├── 02_preprocessing.ipynb                # Text cleaning & geo-resolution
│   ├── 03_feature_engineering.ipynb          # Credibility feature engineering
│   ├── 04_event_classification.ipynb         # XLM-RoBERTa fine-tuning ← CORE
│   ├── 05_credibility_and_clustering.ipynb   # DBSCAN event clustering
│   ├── 06_evaluation_and_explainability.ipynb # SHAP + evaluation
│   └── 07_dashboard_and_map.ipynb            # Interactive map generation
├── data/
│   ├── raw/                                  # Raw collected posts
│   ├── processed/                            # Cleaned & classified posts
│   └── external/                             # Bulk training datasets
├── models/
│   └── crisis_classifier/
│       ├── model_weights/                    # Fine-tuned XLM-RoBERTa weights
│       └── tokenizer/                        # Saved tokenizer
├── outputs/
│   ├── maps/
│   │   └── interactive_crisis_map.html       # Folium interactive map
│   ├── figures/                              # Training curves, confusion matrix
│   └── events_dashboard.geojson             # GeoJSON for mapping
├── src/
│   ├── clustering_utils.py                   # DBSCAN + composite distance
│   └── credibility_utils.py                  # Credibility scoring
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA GPU (recommended for training; CPU works for inference)
- Google Drive (for Colab persistence)

### Install Dependencies

```bash
# Core ML
pip install torch transformers datasets sentence-transformers

# Data & Visualization
pip install pandas numpy scikit-learn shap folium tqdm

# Data Sources
pip install praw requests beautifulsoup4
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Google Colab Setup

All notebooks auto-detect the Colab environment and mount Google Drive:

```python
# Automatically handled in each notebook
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = '/content/drive/MyDrive/10Academy/crisis-detection-system'
```

---

## Running the Pipeline

Run notebooks **in order**. Each notebook saves outputs that the next one reads.

| Step | Notebook | Purpose | Output |
|:----:|----------|---------|--------|
| 1 | `01_data_collection.ipynb` | Collect live posts | `posts_unified.csv` |
| 1B | `01b_bulk_dataset_loader.ipynb` | Load 28K training data | `training_data.csv` |
| 2 | `02_preprocessing.ipynb` | Clean & geo-resolve | `posts_processed.csv` |
| 3 | `03_feature_engineering.ipynb` | Feature engineering | Features added to CSV |
| **4** | **`04_event_classification.ipynb`** | **Train & classify** | `classified_events.csv` + model weights |
| 5 | `05_credibility_and_clustering.ipynb` | Cluster events | `clustered_events.csv` |
| 6 | `06_evaluation_and_explainability.ipynb` | SHAP + metrics | Evaluation report + figures |
| 7 | `07_dashboard_and_map.ipynb` | Interactive map | `interactive_crisis_map.html` |

### Quick Start (Google Colab)

```
1. Open Google Colab
2. Upload notebooks to Drive at:
   MyDrive/10Academy/crisis-detection-system/notebooks/
3. Run NB01B first (one-time training data setup — run only once)
4. Then run sequentially: NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07
```

---

## Model Performance

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `xlm-roberta-base` |
| Task | Sequence Classification (7 classes) |
| Optimizer | AdamW (weight decay = 0.01) |
| Learning Rate | 2e-5 with linear warmup |
| Batch Size | 32 |
| Epochs | 5 |
| Warmup Steps | 373 (10% of total steps) |
| Scheduler | Linear decay with warmup |
| Device | CUDA (NVIDIA T4 GPU — Google Colab) |

### Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1 | 0.568 | 82.2% | 0.170 | 94.8% |
| 2 | 0.138 | 96.1% | 0.130 | 96.4% |
| 3 | 0.099 | 97.3% | 0.148 | 96.4% |
| 4 | 0.071 | 98.1% | 0.142 | 96.5% |
| **5** | **0.052** | **98.7%** | 0.148 | **96.7% ⭐** |

> Best model selected by validation accuracy at epoch 5. No significant overfitting observed — val loss remains stable from epoch 2–5.

### Data Enrichment Impact

| Training Run | Change Made | Val Accuracy | Macro F1 |
|:------------:|------------|:------------:|:--------:|
| Run 1 | Baseline (original datasets) | 95.4% | 0.94 |
| Run 2 | +3,897 earthquake samples (USGS API) | 95.6% | 0.94 |
| Run 3 | +4,621 fire samples (NASA FIRMS + synthetic) | **96.7%** | **0.96** |

---

## Dataset

### Training Data (28,415 samples)

| Source | Samples | Description |
|--------|:-------:|-------------|
| HumAID | ~77K raw → filtered | 19 disaster events, humanitarian tweets (Alam et al., 2021) |
| CrisisLex T6 | ~6K | 6 crisis events from 2013 (Olteanu et al., 2014) |
| DisasterTweets | ~7.6K | Kaggle NLP Getting Started competition |
| **USGS Earthquake API** | ~4K generated | Real M4.5+ seismic events → tweet templates |
| **NASA FIRMS** | ~500 generated | Satellite-detected fires → news-style text |
| **Synthetic** | ~3K | BMKG-style Indonesian & global crisis templates |

### Label Distribution (after balancing to max 5,000/class)

```
flood       5,000  ████████████████████████████████████████
violence    5,000  ████████████████████████████████████████
fire        5,000  ████████████████████████████████████████
other       5,000  ████████████████████████████████████████
accident    4,647  ████████████████████████████████████████
storm       4,422  ████████████████████████████████████████
earthquake  3,966  █████████████████████████████████████
```

### Class Imbalance Fix

Initial datasets severely under-represented two critical classes:

```
earthquake:  69 samples  →  3,966 (+5,649%)   using USGS Earthquake API
fire:       379 samples  →  5,000 (+1,219%)   using NASA FIRMS + templates
```

This was the key insight that pushed accuracy from 95.4% → **96.7%** and both F1 scores to **1.00**.

---

## Technologies Used

| Category | Technology |
|----------|------------|
| **NLP Model** | XLM-RoBERTa-base (Hugging Face Transformers) |
| **Multilingual Embeddings** | SBERT `paraphrase-multilingual-mpnet-base-v2` |
| **Event Clustering** | DBSCAN with composite distance matrix (scikit-learn) |
| **Explainability** | SHAP PartitionExplainer |
| **Geospatial Mapping** | Folium + OpenStreetMap + Leaflet.js |
| **Earthquake Data** | USGS Earthquake Hazards API (free) |
| **Fire Data** | NASA FIRMS MODIS/VIIRS (free) |
| **Official Alerts** | BMKG RSS Feed (Indonesia Met Agency) |
| **Social Media** | Reddit via PRAW API |
| **Training Hardware** | NVIDIA T4 GPU (Google Colab) |
| **Language** | Python 3.10 |

---

## References

1. Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
2. Alam, F., et al. (2021). *HumAID: Human-Annotated Disaster Incidents Data.* ICWSM 2021.
3. Olteanu, A., et al. (2014). *CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises.* ICWSM 2014.
4. Lundberg, S., Lee, S-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
5. Reimers, N., Gurevych, I. (2019). *Sentence-BERT.* EMNLP 2019.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Developed as part of the **10Academy AI Mastery Program — June 2026**.

*Real-Time Crisis Detection System · Built with purpose for humanitarian impact.*