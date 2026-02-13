# Visualising Natural Disaster Image Embeddings

**Author:** Rashid Pandor  
**Supervisor:** XinHui Ma  
**Institution:** University of Hull, 2026

---

## Project Overview

My project explores how unsupervised learning techniques can be applied to crisis imagery for humanitarian response. When natural disasters occur, social media platforms become flooded with millions of images from affected areas. The challenge for humanitarian organisations is finding relevant, actionable images amongst this overwhelming volume of data.

Traditional approaches to image classification require pre-defined labels and training data, which makes them inflexible when new types of crises emerge. This project takes a different approach: using zero-shot semantic search powered by OpenAI's CLIP model, the system allows users to search for images using natural language descriptions without any prior labelling.

The dataset used is CrisisMMD, a publicly available collection of approximately 18,000 disaster images from Twitter, spanning seven major disaster events including Hurricane Harvey, the Mexico Earthquake, and the California Wildfires.

---

## How It Works

The system follows a four-stage pipeline:

### 1. Data Ingestion

The raw CrisisMMD dataset contains some noise: corrupted files, low-resolution images, and extreme aspect ratios. The `clean_data.py` script filters these out, checking each image for:

- Minimum resolution (200x200 pixels)
- File integrity (no corruption)
- Acceptable aspect ratios (between 0.25 and 4.0)
- Minimum file size (5KB)

After cleaning, 17,463 usable images remain across seven disaster events.

### 2. Vectorisation

Each image is passed through the CLIP ViT-B/32 model to generate a 512-dimensional semantic embedding. CLIP was trained on 400 million image-text pairs, so it understands visual concepts without needing domain-specific training. The embeddings are L2-normalised to enable cosine similarity comparisons.

This step is handled by `vectorise.py`, which processes images in batches to manage GPU memory on consumer hardware.

### 3. Dimensionality Reduction

512 dimensions cannot be visualised directly. The `umap_reduction.py` script applies UMAP (Uniform Manifold Approximation and Projection) to reduce the embeddings to 2D coordinates whilst preserving the semantic structure. UMAP was chosen over t-SNE because it is faster on large datasets and better preserves global relationships between clusters.

Parameters used:
- n_neighbours: 15
- min_dist: 0.1
- metric: cosine

### 4. Interactive Interface

The Dash application (`app_demo.py`) provides an interactive visualisation where users can:

- **Explore the embedding space:** Each point on the scatter plot represents an image. Points that are close together are semantically similar.
- **Filter by event:** Isolate images from a specific disaster to see its "semantic footprint" across the embedding space.
- **Semantic search:** Type natural language queries like "flooded street" or "collapsed building" and the system retrieves matching images in real-time.
- **Visual query:** Click any point to find visually similar images using nearest-neighbour search in the embedding space.

---

## Privacy Protection

Working with crisis imagery raises ethical concerns about victim dignity. The application includes an automated face-blurring pipeline that detects faces using Haar Cascade classifiers and applies Gaussian blur before serving images to the browser. This feature is always enabled and cannot be toggled off.

---

## Evaluation

Cluster quality was assessed using standard metrics:

- **Silhouette Score:** -0.11 (indicating overlapping clusters)
- **Davies-Bouldin Index:** Calculated for comparison across configurations
- **Calinski-Harabasz Index:** Measured to assess cluster density

The negative silhouette score is expected and meaningful: disaster imagery is inherently ambiguous. A collapsed building in Mexico looks visually similar to one in Iraq. This overlap validates the hypothesis that rigid classification approaches are unsuitable for this domain, and justifies the need for flexible semantic search.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embeddings | OpenAI CLIP (ViT-B/32) via Hugging Face Transformers |
| Dimensionality Reduction | UMAP |
| Web Framework | Dash (Plotly) with Flask backend |
| Visualisation | Plotly, Scattergl for performance |
| Face Detection | OpenCV Haar Cascades |
| Data Processing | NumPy, Pandas, Pillow |
| Similarity Search | Scikit-learn (cosine similarity) |




Final_Stage_Project_Rashid_LBS-25-986/
├── src/
│   ├── app_demo.py          # Main Dash application
│   ├── app_backend.py       # CLIP model and search logic
│   ├── clean_data.py        # Data cleaning pipeline
│   ├── vectorise.py         # CLIP embedding generation
│   ├── umap_reduction.py    # UMAP dimensionality reduction
│   ├── evaluate_clusters.py # Clustering metrics
│   ├── analytics.py         # Embedding space analytics
│   └── utils/               # Helper modules
├── config/
│   └── settings.py          # Central configuration
├── assets/
│   └── style.css            # Modern Academic theme
├── data/
│   ├── raw/                  # Original CrisisMMD dataset
│   ├── processed/            # Cleaned images
│   ├── embeddings/           # CLIP embeddings
│   └── visualisation/        # UMAP coordinates
└── reports/
    └── metrics/              # Evaluation results
```

---

## Running the Project

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU recommended (runs on CPU but not recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Pipeline Execution

Run each script in order:

```bash
# 1. Clean the raw dataset
python src/clean_data.py

# 2. Generate CLIP embeddings
python src/vectorise.py

# 3. Run UMAP reduction
python src/umap_reduction.py

# 4. (Optional) Evaluate clustering quality
python src/evaluate_clusters.py

# 5. Launch the visualisation!!!
python src/app_demo.py
```

The application will be available at `http://127.0.0.1:8050/`.

---

## Acknowledgements

This project was developed as part of an Honours dissertation under the supervision of XinHui Ma. The CrisisMMD dataset was provided by the Qatar Computing Research Institute.

---

## References

- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- Alam, F. et al. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. ICWSM.
