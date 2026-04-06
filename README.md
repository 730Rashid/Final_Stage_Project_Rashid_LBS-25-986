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

The system follows a multi-stage pipeline:

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

### 4. HDBSCAN Clustering and Auto-Naming

The `cluster_discovery.py` script applies HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to the CLIP embeddings to discover natural groupings without specifying the number of clusters in advance. Each discovered cluster is automatically named by computing cosine similarity between the cluster centroid and a set of candidate text labels encoded with CLIP, assigning the best-matching label to each cluster.

### 5. Interactive Interface

The Dash application (`app_demo.py`) provides an interactive visualisation where users can:

- **Explore the embedding space:** Each point on the scatter plot represents an image. Points that are close together are semantically similar.
- **Filter by event:** Isolate images from a specific disaster to see its "semantic footprint" across the embedding space.
- **Semantic search:** Type natural language queries like "flooded street" or "collapsed building" and the system retrieves matching images in real-time.
- **Multimodal search (Image + Text):** Upload a reference image and optionally add a text refinement (e.g. upload a flooded street photo and type "but with collapsed power lines"). The system encodes the image through CLIP's vision encoder, blends it with the text query embedding using a weighted average, and searches the dataset for the best matches. This enables composite queries that go beyond what either modality can express alone.
- **Visual query:** Click any point to find visually similar images using nearest-neighbour search in the embedding space.
- **Zero-shot classification:** Each image card displays classification badges (e.g. Flood, Fire, Debris, Rescue) with confidence scores, computed using CLIP text-image cosine similarity against 10 disaster-related labels.
- **Damage severity scoring:** A colour-coded severity badge (Critical, Severe, Moderate, Minimal) on each image card, computed using a two-anchor contrast approach comparing each image to "catastrophic damage" vs "undamaged" text embeddings with a sigmoid-scaled score.
- **CLIP captioning:** Automatically generated natural language captions describing the scene content, using CLIP interrogation across scene, damage, object, and weather categories.
- **Attention heatmaps:** Explainable AI visualisations showing which spatial regions CLIP's vision encoder attends to most when encoding an image, computed via patch-CLS cosine similarity on ViT-B/32 hidden states.
- **Embedding analytics:** Six analytics tabs including event statistics, embedding space analysis, cross-disaster transfer, topological analysis, multi-model comparison, and data export.

---

## Privacy Protection

Working with crisis imagery raises ethical concerns about victim dignity. The application includes an automated face-blurring pipeline using YuNet, a lightweight CNN-based deep learning face detector (228KB ONNX model) built into OpenCV. YuNet provides significantly higher accuracy than traditional Haar cascades while maintaining real-time performance (30-100ms per image). A Haar cascade fallback is retained for environments where the YuNet model is unavailable. Face blurring is always enabled and cannot be toggled off.

---

## Explainable AI: Attention Heatmaps

The system generates spatial relevance heatmaps for each image by measuring the cosine similarity between each of the 49 spatial patch hidden states and the CLS token hidden state in CLIP's ViT-B/32 vision encoder. Patches with high similarity to the CLS token contributed most strongly to the global image representation, revealing where the model focused its attention.

This approach is robust to all HuggingFace attention implementations (eager, SDPA, Flash Attention) since it only requires the final hidden states, not raw attention weight tensors. The resulting 7x7 relevance grid is upsampled to the original image dimensions with bicubic interpolation, coloured with a JET colourmap, and alpha-blended onto the original image.

---

## Topological Data Analysis

The Topology tab applies two techniques from algebraic topology and discrete geometry to characterise the global structure of the CLIP embedding space:

### Persistent Homology (Vietoris-Rips Filtration)

A Vietoris-Rips filtration is constructed on a random subsample of 1,995 CLIP embeddings using cosine distance. The filtration tracks topological features across increasing distance thresholds:

- **H0 (connected components):** 1,990 features detected, representing distinct semantic clusters that merge as the distance threshold increases.
- **H1 (loops/cycles):** 1,772 features detected, revealing topological holes in the embedding space where regions of semantic ambiguity exist between disaster categories.

The persistence diagram visualises the birth-death pairs for each feature, with long-lived features indicating robust topological structure rather than noise.

### Ollivier-Ricci Curvature

A 10-nearest-neighbour graph is constructed on the same subsample, and Ollivier-Ricci curvature is computed for each edge using optimal transport (Earth Mover's Distance). The curvature reveals local geometry:

- **Positive curvature (mean = 0.320):** Dense, sphere-like cluster regions where images share strong semantic similarity.
- **Negative curvature (7.3% of edges):** Bridge edges connecting semantically distant regions, identifying ambiguous images that sit between disaster categories.

Per-event curvature analysis shows how tightly each disaster's images cluster in the embedding space.

---

## Multi-Model Embedding Comparison (Ablation Study)

The Model Comparison tab provides a systematic ablation study comparing three embedding models across three dimensionality reduction methods and two clustering algorithms:

### Models

| Model | Dimensions | Training Data | Architecture |
|-------|-----------|---------------|--------------|
| **CLIP ViT-B/32** | 512 | 400M image-text pairs (OpenAI) | Vision Transformer |
| **SigLIP-base** | 768 | 10B image-text pairs (Google WebLI) | Vision Transformer (sigmoid loss) |
| **ResNet50** | 2,048 | ImageNet (1.2M images, no text) | CNN baseline |

### Reduction Methods

- **UMAP** (n_neighbours=15, min_dist=0.1, cosine metric)
- **t-SNE** (perplexity=30, 200 iterations, euclidean metric on L2-normalised embeddings)
- **PCA** (2 components, whitened)

### Clustering Algorithms

- **HDBSCAN** (min_cluster_size=50, density-based, no predefined k)
- **K-Means** (k=7, matching the number of disaster events)

### Evaluation Metrics

- **Silhouette Score:** Measures cluster separation (-1 to 1, higher is better)
- **Davies-Bouldin Index:** Measures inter-cluster similarity (lower is better)
- **Calinski-Harabasz Index:** Measures cluster density (higher is better)
- **Trustworthiness:** Whether low-dimensional neighbours were also near in the original space (0 to 1)
- **Continuity:** Whether original-space neighbours are preserved in the reduction (0 to 1)

The 3x3 scatter grid visualises all 9 model-reduction combinations, with a subsampled 2,000 points per plot for performance. Trustworthiness and continuity are computed on a stratified subsample of 3,000 images to avoid O(N^2) memory requirements.

The pipeline is run offline by `vectorise_models.py` (embedding generation) and `model_comparison.py` (reduction, clustering, and metric evaluation), with results cached to `comparison_results.json`.

---

## Damage Severity Scoring

Each image receives a damage severity score using a two-anchor contrast approach:

1. Two text anchors are encoded with CLIP: one describing catastrophic damage and one describing intact/undamaged scenes
2. The cosine similarity between each image embedding and both anchors is computed
3. The contrast (disaster_similarity - normal_similarity) is passed through a sigmoid function with temperature 20
4. The resulting score (0-1) is mapped to a severity category: Critical (75%+), Severe (50-75%), Moderate (25-50%), or Minimal (0-25%)

This zero-shot approach requires no labelled training data and provides direct humanitarian value for rapid damage assessment.

---

## Evaluation

Cluster quality was assessed using standard metrics:

- **Silhouette Score:** -0.11 (indicating overlapping clusters)
- **Davies-Bouldin Index:** Calculated for comparison across configurations
- **Calinski-Harabasz Index:** Measured to assess cluster density

The negative silhouette score is expected and meaningful: disaster imagery is inherently ambiguous. A collapsed building in Mexico looks visually similar to one in Iraq. This overlap validates the hypothesis that rigid classification approaches are unsuitable for this domain, and justifies the need for flexible semantic search.

---

## Browser Rendering Performance

Visualising 17,463 points interactively in a browser requires careful optimisation. The following techniques are applied to keep the interface responsive:

- **WebGL rendering:** All scatter plots on the main UMAP view use Plotly's `Scattergl` trace type, which renders via WebGL rather than SVG. This offloads point drawing to the GPU and handles 17K+ points smoothly.
- **Hover distance limiting:** `hoverdistance` is set to 20px so the browser only searches nearby points on mousemove rather than scanning all 17K points per frame.
- **UI revision persistence:** `uirevision="constant"` preserves the user's zoom and pan state across callback updates, avoiding expensive full-figure re-renders when filters or searches change.
- **Reduced pixel ratio:** `plotGlPixelRatio` is set to 1 to halve the number of pixels the GPU must render on high-DPI displays, with no visible quality loss on scatter plots.
- **Scroll zoom:** Enabled natively so users can zoom without switching mode bar tools, reducing interaction friction.
- **Zero-copy filtering:** The main callback filters the dataframe using pandas boolean indexing (views) rather than copying the full 17K-row dataframe on every interaction.
- **NumPy array pass-through:** Trace data is passed as `.values` (NumPy arrays) rather than pandas Series, avoiding repeated index alignment overhead inside Plotly.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Primary Embeddings | OpenAI CLIP (ViT-B/32) via Hugging Face Transformers |
| Comparison Models | SigLIP-base (Google), ResNet50 (ImageNet) |
| Dimensionality Reduction | UMAP, t-SNE, PCA |
| Clustering | HDBSCAN, K-Means |
| Topological Analysis | Ripser (persistent homology), POT (optimal transport) |
| Web Framework | Dash (Plotly) with Flask backend |
| Visualisation | Plotly |
| Face Detection | YuNet (DNN, primary), OpenCV Haar Cascades (fallback) |
| Data Processing | NumPy, Pandas, Pillow |
| Similarity Search | Scikit-learn (cosine similarity) |
| Heatmaps | OpenCV (JET colourmap, alpha blending) |
| GPU Acceleration | PyTorch with CUDA |

---

## Project Structure

```
Final_Stage_Project_Rashid_LBS-25-986/
├── src/
│   ├── app_demo.py            # Main Dash application and UI (6 analytics tabs)
│   ├── app_backend.py         # CLIP model, search, classification, severity scoring
│   ├── clean_data.py          # Data cleaning pipeline
│   ├── vectorise.py           # CLIP embedding generation
│   ├── vectorise_models.py    # SigLIP and ResNet50 embedding generation
│   ├── umap_reduction.py      # UMAP dimensionality reduction
│   ├── model_comparison.py    # Multi-model ablation study pipeline
│   ├── cluster_discovery.py   # HDBSCAN clustering and auto-naming
│   ├── evaluate_clusters.py   # Clustering metrics
│   ├── analytics.py           # Embedding space analytics
│   ├── topology_analysis.py   # Persistent homology and Ollivier-Ricci curvature
│   ├── clip_captioning.py     # CLIP interrogation for image captioning
│   ├── clip_heatmaps.py       # Patch-CLS attention heatmaps (Explainable AI)
│   └── utils/
│       ├── event_utils.py     # Event parsing and mapping
│       ├── file_utils.py      # File handling utilities
│       └── gpu_utils.py       # GPU memory management
├── config/
│   ├── settings.py            # Central configuration
│   └── logging_config.py      # Logging setup
├── assets/
│   └── style.css              # Modern Academic theme
├── data/
│   ├── raw/                   # Original CrisisMMD dataset
│   ├── processed/             # Cleaned images
│   ├── embeddings/            # CLIP embeddings (.npy)
│   ├── visualisation/         # UMAP coordinates (.json)
│   ├── comparison/            # Multi-model embeddings and results
│   ├── models/                # YuNet face detection model (.onnx)
│   └── cache/                 # Blurred image cache
├── tests/
│   └── test_unit_report.py    # Unit tests (UT-01 to UT-05 from Table 5)
└── reports/
    └── metrics/               # Evaluation results
```

---

## Running the Project

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU recommended (runs on CPU but slower)

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

# 4. (Optional) Run HDBSCAN clustering
python src/cluster_discovery.py

# 5. (Optional) Evaluate clustering quality
python src/evaluate_clusters.py

# 6. (Optional) Generate SigLIP and ResNet50 embeddings for ablation study
python src/vectorise_models.py

# 7. (Optional) Run multi-model comparison pipeline
python src/model_comparison.py

# 8. Launch the visualisation
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
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD.
- Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL.
- Edelsbrunner, H. & Harer, J. (2010). Computational Topology: An Introduction. AMS.
- Ollivier, Y. (2009). Ricci Curvature of Markov Chains on Metric Spaces. Journal of Functional Analysis.
- Zhai, X. et al. (2023). Sigmoid Loss for Language Image Pre-Training. ICCV.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
