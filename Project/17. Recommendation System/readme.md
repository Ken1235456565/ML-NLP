# Music Recommender System

A full-stack personalized music recommendation system built on the [Million Song Dataset (MSD)](http://millionsongdataset.com/), covering the complete pipeline from data ingestion to online serving — including multi-source recall, learning-to-rank, cold start handling, and A/B testing.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Details](#pipeline-details)
- [Models](#models)
- [Evaluation](#evaluation)
- [Cold Start](#cold-start)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

---

## Overview

This system recommends music to users by combining **collaborative filtering**, **content-based retrieval**, and **learning-to-rank** into a classic two-stage recall-and-rank architecture. It is designed to handle the scale of the Million Song Dataset (1M songs, 48M user play events) while keeping online latency under 50ms.

**Key capabilities:**

- Multi-source recall: ItemCF, UserCF, ALS matrix factorization, and audio-feature-based content recall
- Ranking: Logistic Regression → GBDT → FM → DNN+GBDT+FM ensemble
- Cold start strategies for both new users and new songs
- Online A/B testing with orthogonal traffic splitting across pipeline layers
- REST API powered by FastAPI + Redis feature store

---

## Dataset

This project is built on the **Million Song Dataset** and its companion datasets:

| Dataset | Description | Size |
|---|---|---|
| MSD Core (HDF5) | Audio features & metadata for 1M songs (55 fields per song) | 280 GB |
| MSD Subset | 10,000-song subset for quick experiments | 1.8 GB |
| Taste Profile | User–song play counts for 1M+ users | 48M rows |
| MusiXmatch Lyrics | Bag-of-words lyrics for ~237K songs | — |
| MAGD Genre | Genre labels across 21 categories | — |

**Key audio features used:**

- **Scalar:** `tempo`, `loudness`, `duration`, `key`, `mode`, `danceability`, `energy`
- **Sequence:** `segments_timbre` (N×12 MFCC-like), `segments_pitches` (N×12)
- **Metadata:** `artist_name`, `title`, `year`, `artist_terms` (genre/style tags)

Download instructions: [millionsongdataset.com](http://millionsongdataset.com/)

> **Tip:** Start with the 10K subset to validate your pipeline before scaling to the full dataset.

---

## System Architecture

```
                        ┌─────────────────────────────────┐
                        │         Offline Pipeline         │
                        │                                  │
  Raw MSD Data ──────▶  │  Data Ingestion                  │
  (HDF5 + TSV)          │       │                          │
                        │       ▼                          │
                        │  Feature Engineering             │
                        │       │                          │
                        │       ▼                          │
                        │  Model Training                  │
                        │  (ALS / FM / DNN)                │
                        │       │                          │
                        │       ▼                          │
                        │  Embeddings & Model Artifacts ───┼──▶ Redis / Faiss
                        └─────────────────────────────────┘
                                                                     │
                        ┌─────────────────────────────────┐          │
  User Request ───────▶ │        Online Serving            │◀─────────┘
                        │                                  │
                        │  A/B Test Router                 │
                        │       │                          │
                        │       ▼                          │
                        │  Multi-Source Recall             │
                        │  (ItemCF + ALS + Content)        │
                        │       │                          │
                        │       ▼                          │
                        │  FM / DNN Ranking                │
                        │       │                          │
                        │       ▼                          │
                        │  Re-rank (Diversity + Rules)     │
                        │       │                          │
                        └───────┼─────────────────────────┘
                                │
                                ▼
                        Top-N Song List
```

---

## Project Structure

```
music_recommender/
│
├── data/
│   ├── raw/                    # Original MSD data files
│   │   ├── hdf5/               # Per-song HDF5 files (55 fields each)
│   │   ├── tasteprofile/       # User play history (user_id, song_id, play_count)
│   │   ├── genre/              # Genre annotations (msd-MAGD-genreAssignment.tsv)
│   │   └── mxm_lyrics/         # MusiXmatch lyrics (bag-of-words)
│   ├── processed/              # Cleaned and transformed data
│   │   ├── songs_features.parquet
│   │   ├── user_song_matrix.parquet
│   │   └── train/ val/ test/
│   └── online/                 # Live serving assets
│       ├── song_embeddings.npy
│       └── user_embeddings.npy
│
├── src/
│   ├── data_pipeline/          # Ingestion, cleaning, feature engineering
│   ├── recall/                 # ItemCF, UserCF, ALS, content-based
│   ├── ranking/                # LR, GBDT, FM, DNN rankers
│   ├── models/                 # Model definitions (PyTorch / TensorFlow)
│   ├── serving/                # FastAPI service + recall/rank/rerank layers
│   ├── evaluation/             # Offline metrics + A/B test routing
│   └── cold_start/             # New user & new song strategies
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_experiments.ipynb
│
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── serving_config.yaml
│
├── tests/
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourname/music-recommender.git
cd music-recommender
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Download the 10K subset for local development
wget http://millionsongdataset.com/sites/default/files/AdditionalFiles/subset_msd.tar.gz
tar -xzf subset_msd.tar.gz -C data/raw/hdf5/

# Download Taste Profile
wget http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip
unzip train_triplets.txt.zip -d data/raw/tasteprofile/
```

### 3. Run the data pipeline

```bash
# Extract features from HDF5 files and build the song feature table
python -m src.data_pipeline.hdf5_reader

# Clean data, build train/val/test splits, construct pos/neg samples
python -m src.data_pipeline.data_cleaner
python -m src.data_pipeline.sample_builder
```

### 4. Train recall models

```bash
# Train ItemCF
python -m src.recall.itemcf --config configs/model_config.yaml

# Train ALS with Spark (recommended for full dataset)
spark-submit src/recall/matrix_factorization.py --config configs/model_config.yaml
```

### 5. Train the ranking model

```bash
python -m src.ranking.fm_ranker --config configs/model_config.yaml
```

### 6. Launch the API server

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Pipeline Details

### Data Ingestion

Each song in MSD is stored as an individual HDF5 file. The ingestion pipeline recursively scans the directory tree, extracts the 55 fields, and aggregates them into a single Parquet feature table.

Sequence features such as `segments_timbre` (shape N×12) are summarized into fixed-length vectors using per-column **mean** and **standard deviation** before being used as model input.

### Feature Engineering

**Song-side features:**

| Feature | Type | Notes |
|---|---|---|
| tempo, loudness, duration | Scalar | Min-max normalized |
| danceability, energy | Scalar | Min-max normalized |
| key, mode | Categorical | One-hot encoded |
| timbre_mean / timbre_std | Dense vector (12-dim each) | Aggregated from segments |
| artist_terms | Sparse | Genre/style tags |
| decade | Categorical | Bucketed from year |

**User-side features:**

| Feature | Type | Notes |
|---|---|---|
| total_plays, unique_songs | Scalar | Listening activity stats |
| avg_play, max_play | Scalar | Play depth indicators |
| top_genres | Sparse | Weighted by log play count |

### Positive & Negative Sample Construction

- **Positive samples:** songs the user has played (label = 1)
- **Negative samples:** randomly sampled unplayed songs, balanced 1:1 with positives
- Play counts are log-smoothed: `weight = log(1 + play_count)`

---

## Models

### Recall Layer

| Model | Algorithm | Use Case |
|---|---|---|
| **ItemCF** | Song co-occurrence with popularity penalty | Main recall for active users |
| **UserCF** | User similarity via inverted index | Social / taste-similar users |
| **ALS** | Distributed matrix factorization (Spark) | Full 48M-row Taste Profile |
| **Content-Based** | Faiss ANN on audio feature vectors | New songs, currently-playing context |

### Ranking Layer

Models are applied sequentially — start with LR as a baseline and progress toward the full ensemble:

```
LR  →  GBDT  →  GBDT+LR  →  GBDT+FM  →  DNN+GBDT+FM
```

The **DNN+GBDT+FM ensemble** is the production model:
- **DNN** extracts deep representations from dense/embedding features
- **GBDT** provides nonlinear feature transformations as leaf-node features
- **FM** serves as the fusion layer, handling all sparse feature interactions with O(n) complexity

### Re-ranking

After scoring, a rule-based re-ranker applies:
1. **Genre diversity** — no more than 3 consecutive songs from the same genre
2. **Blacklist filtering** — removes unlicensed or flagged tracks (via Redis)
3. **Promotion injection** — inserts 1 promoted track per every 10 organic results

---

## Evaluation

### Offline Metrics

```bash
python -m src.evaluation.offline_metrics --model itemcf --split test
```

Supported metrics: `Precision@K`, `Recall@K`, `NDCG@K`, `Coverage`, `Diversity`

| Model | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| ItemCF (baseline) | — | — | — |
| ALS | — | — | — |
| GBDT+FM | — | — | — |
| DNN+GBDT+FM | — | — | — |

> Fill in your results after running experiments.

### Online A/B Testing

Traffic is split deterministically by hashing `layer:user_id`, ensuring the same user always enters the same experiment bucket and that different pipeline layers remain **orthogonal**.

```yaml
# configs/model_config.yaml
ab_experiments:
  exp_als_v2:
    traffic: 0.10
    model: als_v2
  exp_fm:
    traffic: 0.10
    model: fm
  control:
    traffic: 0.80
    model: itemcf_base
```

Key online metrics: **CTR**, **listen-through rate**, **session length**, **7-day retention**.

---

## Cold Start

### New User

| Signal Available | Strategy |
|---|---|
| None | Popularity-based recommendations by global trending |
| Genre selection (onboarding) | Per-genre hot songs, instant personalization |
| First played song | Content-based expansion from seed song via Faiss |

### New Song

1. Extract audio features from the HDF5 file upon ingestion
2. Find the most similar existing songs via content-based recall (Faiss ANN)
3. Target the historical listeners of those similar songs as the initial audience
4. Gradually incorporate real play-count feedback as the song accumulates interactions

---

## API Reference

### `POST /recommend`

Returns a ranked list of song recommendations for a given user.

**Request body:**

```json
{
  "user_id": "user_abc123",
  "scene": "homepage",
  "current_song": null,
  "topk": 30
}
```

`scene` options: `homepage` | `playing` | `search`

**Response:**

```json
{
  "user_id": "user_abc123",
  "songs": [
    {"song_id": "SOXXXXX", "score": 0.923},
    {"song_id": "SOYYYYY", "score": 0.887}
  ],
  "latency_ms": 18.4
}
```

**Target latency:** < 50ms (p99)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | PySpark, Pandas, h5py |
| Vector search | Faiss |
| ML models | PyTorch, Scikit-learn, LightGBM |
| Deep learning | TensorFlow / PyTorch |
| Feature store | Redis |
| API server | FastAPI, Uvicorn |
| Experiment tracking | MLflow (optional) |
| Notebooks | Jupyter |

---

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please make sure all new code includes unit tests under `tests/` and passes `pytest` before submitting.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The Million Song Dataset is provided for research purposes. Please refer to the [MSD website](http://millionsongdataset.com/) for its specific terms of use.

---

## Citation

If you use this project in your research, please cite the Million Song Dataset:

```bibtex
@inproceedings{Bertin-Mahieux2011,
  author    = {Thierry Bertin-Mahieux and Daniel P.W. Ellis and
               Brian Whitman and Paul Lamere},
  title     = {The Million Song Dataset},
  booktitle = {Proceedings of the 12th International Society
               for Music Information Retrieval Conference (ISMIR)},
  year      = {2011}
}
```
