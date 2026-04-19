# Visualising Natural Disaster Image Embeddings

Author: Rashid Pandor
Supervisor: XinHui Ma
Institution: University of Hull, 2026

## Project Overview

This project explores how unsupervised learning can be applied to crisis imagery for humanitarian response. When a natural disaster hits, social media floods with millions of images from affected areas, and the real problem for humanitarian organisations is finding the relevant ones fast enough for that information to still matter.

Traditional classifiers need predefined labels and training data, which gets awkward the moment a new type of crisis appears. This project takes a different route. It uses zero shot semantic search powered by OpenAI's CLIP model, so users can search with ordinary language without anyone having to label the dataset first.

The data comes from CrisisMMD, a public collection of roughly 18,000 disaster images pulled from Twitter, covering seven major events including Hurricane Harvey, the Mexico Earthquake, and the California Wildfires.

## How It Works

The system is a pipeline of a few stages, each one feeding the next.

### 1. Data Ingestion

The raw CrisisMMD dataset is a bit noisy. There are corrupted files, tiny thumbnails, and images with extreme aspect ratios that would throw off the model. The `clean_data.py` script handles the filtering, checking each image for minimum resolution (200 by 200 pixels), file integrity, acceptable aspect ratio (between 0.25 and 4.0), and a minimum file size of 5KB. After cleaning, 17,463 usable images survive across seven disaster events.

### 2. Vectorisation

Each image is pushed through the CLIP ViT B/32 model to produce a 512 dimensional semantic embedding. CLIP was trained on 400 million image and text pairs, so it already has a reasonable grasp of visual concepts without any domain specific tuning. The embeddings are L2 normalised so that cosine similarity works cleanly later on. `vectorise.py` handles this in batches to stay within GPU memory limits on consumer hardware.

### 3. Dimensionality Reduction

You cannot visualise 512 dimensions directly, so `umap_reduction.py` takes the embeddings down to 2D coordinates using UMAP. I picked UMAP over t SNE mainly because it scales better on large datasets and tends to preserve the overall structure between clusters, not just local neighbourhoods. The parameters used are n_neighbours 15, min_dist 0.1, and cosine metric.

### 4. HDBSCAN Clustering and Auto Naming

`cluster_discovery.py` runs HDBSCAN over the CLIP embeddings to find natural groupings without having to specify how many clusters there should be. Each discovered cluster is then labelled automatically: the cluster centroid is compared via cosine similarity to a pool of candidate text labels encoded with CLIP, and the best matching label wins.

### 5. Interactive Interface

The Dash application (`app_demo.py`) is where everything comes together.

Explore the embedding space. Each point is an image. Points close together are semantically similar.

Filter by event. Isolate images from a single disaster to see its semantic footprint across the embedding space.

Semantic search. Type something like "flooded street" or "collapsed building" and the system retrieves matching images in real time.

Multimodal search (image plus text). Upload a reference image and optionally add a text refinement such as "but with collapsed power lines". The system encodes the image through CLIP's vision encoder, blends the result with the text query embedding using a weighted average, and searches the dataset for the best matches. This is useful for composite queries that neither modality alone can really express.

Visual query. Click any point on the scatter plot and the system runs nearest neighbour search in the embedding space, pulling up everything that looks like the anchor image. Handy for building a gallery of one specific kind of damage.

Zero shot classification. Every image card shows classification badges (Flood, Fire, Debris, Rescue, and so on) with confidence scores, computed from CLIP text image cosine similarity against ten disaster related labels.

Damage severity scoring. A colour coded severity badge (Critical, Severe, Moderate, Minimal) appears on each card, using a two anchor contrast approach comparing each image to "catastrophic damage" and "undamaged" text embeddings with a sigmoid scaled score.

CLIP captioning. Natural language captions are produced automatically using CLIP interrogation across scene, damage, object, and weather categories.

Attention heatmaps. Explainable AI visualisations showing which spatial regions CLIP's vision encoder is paying attention to when encoding an image, computed via patch CLS cosine similarity on ViT B/32 hidden states.

Embedding analytics. Six analytics tabs covering event statistics, embedding space analysis, cross disaster transfer, topological analysis, multi model comparison, and data export.

## Privacy Protection

Crisis imagery raises a real ethical question about victim dignity. The application runs an automated face blurring pipeline using YuNet, a lightweight CNN based face detector (228KB ONNX model) built into OpenCV. YuNet is quite a bit more accurate than the older Haar cascade approach while still running in 30 to 100 ms per image, which is fast enough for the demo. A Haar cascade fallback is kept for environments where the YuNet model cannot be loaded. Face blurring is always on and cannot be toggled off.

## Explainable AI: Attention Heatmaps

The system produces a spatial relevance heatmap for each image by measuring cosine similarity between each of the 49 spatial patch hidden states and the CLS token hidden state in CLIP's ViT B/32 vision encoder. Patches with high similarity to the CLS token are the ones that contributed most to the global image representation, which is a decent proxy for "where did the model look".

This approach plays nicely with every HuggingFace attention implementation (eager, SDPA, Flash Attention) because it only needs the final hidden states, not raw attention weight tensors. The resulting 7 by 7 relevance grid is upsampled to the original image dimensions using bicubic interpolation, coloured with a JET colourmap, and alpha blended over the original image.

## Topological Data Analysis

The Topology tab applies two techniques from algebraic topology and discrete geometry to characterise the global structure of the CLIP embedding space.

### Persistent Homology (Vietoris Rips Filtration)

A Vietoris Rips filtration is built on a random subsample of 1,995 CLIP embeddings using cosine distance. The filtration tracks topological features as the distance threshold grows.

H0 (connected components): 1,990 features detected, representing distinct semantic clusters that merge as the threshold grows.

H1 (loops or cycles): 1,772 features detected, which corresponds to topological holes in the embedding space. These show up where there is semantic ambiguity between disaster categories.

The persistence diagram plots birth and death pairs for each feature. Long lived features tend to indicate genuine topological structure rather than noise.

### Ollivier Ricci Curvature

A 10 nearest neighbour graph is constructed on the same subsample, and Ollivier Ricci curvature is computed per edge using optimal transport (Earth Mover's Distance). The result reveals local geometry.

Positive curvature (mean 0.320): dense, sphere like cluster regions where images share strong semantic similarity.

Negative curvature (7.3 percent of edges): bridge edges connecting semantically distant regions, marking ambiguous images that sit between categories.

Per event curvature also shows how tightly each disaster's images cluster in the embedding space.

## Multi Model Embedding Comparison (Ablation Study)

The Model Comparison tab runs a systematic ablation comparing three embedding models across three reduction methods and two clustering algorithms.

### Models

| Model | Dimensions | Training Data | Architecture |
|-------|-----------|---------------|--------------|
| CLIP ViT B/32 | 512 | 400M image text pairs (OpenAI) | Vision Transformer |
| SigLIP base | 768 | 10B image text pairs (Google WebLI) | Vision Transformer (sigmoid loss) |
| ResNet50 | 2,048 | ImageNet (1.2M images, no text) | CNN baseline |

### Reduction Methods

UMAP (n_neighbours 15, min_dist 0.1, cosine metric)
t SNE (perplexity 30, 200 iterations, euclidean on L2 normalised embeddings)
PCA (2 components, whitened)

### Clustering Algorithms

HDBSCAN (min_cluster_size 50, density based, no predefined k)
K Means (k 7, matching the number of disaster events)

### Evaluation Metrics

Silhouette score measures cluster separation (range minus 1 to 1, higher is better).
Davies Bouldin index measures inter cluster similarity (lower is better).
Calinski Harabasz index measures cluster density (higher is better).
Trustworthiness checks whether low dimensional neighbours were also nearby in the original space (0 to 1).
Continuity checks whether the original space neighbours are preserved in the reduction (0 to 1).

The 3 by 3 scatter grid visualises all nine model and reduction combinations, with 2,000 subsampled points per plot for performance. Trustworthiness and continuity are computed on a stratified subsample of 3,000 images to avoid O(N squared) memory blowups.

The offline side of this pipeline lives in `vectorise_models.py` (embedding generation) and `model_comparison.py` (reduction, clustering, metrics). Results are cached to `comparison_results.json`.

## Damage Severity Scoring

Every image gets a damage severity score using a two anchor contrast approach.

Two text anchors are encoded with CLIP. One describes catastrophic damage, the other describes intact or undamaged scenes. Cosine similarity is then computed between each image embedding and both anchors. The contrast value (disaster similarity minus normal similarity) is passed through a sigmoid function with temperature 20, and the resulting score from 0 to 1 is mapped to a category: Critical (75 percent and above), Severe (50 to 75), Moderate (25 to 50), or Minimal (below 25).

Zero shot again. No labelled training data required, and it has direct humanitarian value for rapid damage assessment.

## Evaluation

Cluster quality was evaluated using standard metrics.

Silhouette score came out at minus 0.11, which indicates overlapping clusters.
Davies Bouldin index was calculated for comparison across configurations.
Calinski Harabasz index was measured to assess cluster density.

The negative silhouette score is actually expected and is one of the more interesting findings of the project. Disaster imagery is inherently ambiguous: a collapsed building in Mexico often looks a lot like a collapsed building in Iraq. That overlap is exactly the reason rigid classification approaches fall short here, and it makes the case for flexible semantic search as the better tool for this domain.

## Browser Rendering Performance

Rendering 17,463 points interactively in a browser takes some care. The techniques below are what keeps the interface responsive.

WebGL rendering. Every scatter plot on the main UMAP view uses Plotly's Scattergl trace type, which renders through WebGL instead of SVG. This offloads drawing to the GPU and handles 17K or more points smoothly.

Hover distance limiting. `hoverdistance` is capped at 20 pixels so the browser only searches nearby points on mouse move rather than scanning all 17K per frame.

UI revision persistence. `uirevision` is set to a constant so that the user's zoom and pan state is preserved across callback updates, avoiding expensive full figure re renders when filters or searches change.

Reduced pixel ratio. `plotGlPixelRatio` is set to 1 to halve the number of pixels the GPU has to render on high DPI displays. You cannot tell the difference visually on scatter plots.

Scroll zoom. Enabled natively so users can zoom without switching mode bar tools.

Zero copy filtering. The main callback filters the dataframe using pandas boolean indexing (views) rather than copying all 17K rows on every interaction.

NumPy array pass through. Trace data is passed as `.values` (NumPy arrays) rather than pandas Series, which avoids repeated index alignment inside Plotly.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Primary Embeddings | OpenAI CLIP (ViT B/32) via Hugging Face Transformers |
| Comparison Models | SigLIP base (Google), ResNet50 (ImageNet) |
| Dimensionality Reduction | UMAP, t SNE, PCA |
| Clustering | HDBSCAN, K Means |
| Topological Analysis | Ripser (persistent homology), POT (optimal transport) |
| Web Framework | Dash (Plotly) with Flask backend |
| Visualisation | Plotly |
| Face Detection | YuNet (DNN, primary), OpenCV Haar Cascades (fallback) |
| Data Processing | NumPy, Pandas, Pillow |
| Similarity Search | Scikit learn (cosine similarity) |
| Heatmaps | OpenCV (JET colourmap, alpha blending) |
| GPU Acceleration | PyTorch with CUDA |

## Project Structure

```
Final_Stage_Project_Rashid_LBS 25 986/
├── src/
│   ├── app_demo.py            Main Dash application and UI (6 analytics tabs)
│   ├── app_backend.py         CLIP model, search, classification, severity scoring
│   ├── clean_data.py          Data cleaning pipeline
│   ├── vectorise.py           CLIP embedding generation
│   ├── vectorise_models.py    SigLIP and ResNet50 embedding generation
│   ├── umap_reduction.py      UMAP dimensionality reduction
│   ├── model_comparison.py    Multi model ablation study pipeline
│   ├── cluster_discovery.py   HDBSCAN clustering and auto naming
│   ├── evaluate_clusters.py   Clustering metrics
│   ├── analytics.py           Embedding space analytics
│   ├── topology_analysis.py   Persistent homology and Ollivier Ricci curvature
│   ├── clip_captioning.py     CLIP interrogation for image captioning
│   ├── clip_heatmaps.py       Patch CLS attention heatmaps
│   └── utils/
│       ├── event_utils.py     Event parsing and mapping
│       ├── file_utils.py      File handling utilities
│       └── gpu_utils.py       GPU memory management
├── config/
│   ├── settings.py            Central configuration
│   └── logging_config.py      Logging setup
├── assets/
│   └── style.css              Modern academic theme
├── data/
│   ├── raw/                   Original CrisisMMD dataset
│   ├── processed/             Cleaned images
│   ├── embeddings/            CLIP embeddings (.npy)
│   ├── visualisation/         UMAP coordinates (.json)
│   ├── comparison/            Multi model embeddings and results
│   ├── models/                YuNet face detection model (.onnx)
│   └── cache/                 Blurred image cache
├── tests/
│   └── test_unit_report.py    Unit tests (UT 01 to UT 05 from Table 5)
└── reports/
    └── metrics/               Evaluation results
```

## Running the Project

### Prerequisites

Python 3.9 or higher.
At least 16 GB of RAM (the CLIP model plus the 17K embeddings need room to live in memory).
On Windows, a pagefile of at least 16 GB if you plan to run in CPU mode. More on that below.
A CUDA capable GPU with 4 GB VRAM or more is recommended. The app will run on CPU but it is slower.
A browser with WebGL enabled (Chrome, Edge, or Firefox). The scatter plot needs it.

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

Once the offline pipeline has been run, launching the visualisation is a one liner.

```bash
python src/app_demo.py
```

The app will be at `http://127.0.0.1:8050/`.

### Forcing CPU Mode

If your GPU has less than 4 GB VRAM or is not available, set the `FORCE_CPU` environment variable before launching. This bypasses CUDA detection and keeps CLIP on the CPU.

Git Bash, Linux, or macOS:

```bash
export FORCE_CPU=1
python src/app_demo.py
```

Windows Command Prompt:

```cmd
set FORCE_CPU=1
python src/app_demo.py
```

### Pagefile Note (Windows, CPU Mode)

One thing that tripped me up during development. Running CLIP on CPU with the full 17K embedding dataset in memory needs a commit limit of around 20 GB or more. If your machine has 16 GB RAM and the default 4 GB Windows pagefile, the CLIP model load will fail with `DefaultCPUAllocator: not enough memory`.

To fix this, increase the pagefile.

1. Press Win plus R, type `sysdm.cpl`, hit Enter.
2. Go to the Advanced tab, click Settings under Performance, then Advanced, then Change under Virtual Memory.
3. Uncheck "Automatically manage paging file size for all drives".
4. Select the C drive, pick Custom size, and set Initial to 16384 and Maximum to 32768.
5. Click Set. This is the step that everyone misses. Without clicking Set, nothing is saved.
6. Click OK on all the dialogs.
7. Reboot.

You can verify it worked by running `wmic pagefile list /format:list`. `AllocatedBaseSize` should now read 16384.

### Pipeline Execution (Offline)

The offline preprocessing pipeline generates the embeddings, UMAP coordinates, and clustering artefacts used by the live app. These scripts only need to be run once because their outputs are cached under `data/`.

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

# 6. (Optional) Generate SigLIP and ResNet50 embeddings for ablation
python src/vectorise_models.py

# 7. (Optional) Run multi model comparison pipeline
python src/model_comparison.py

# 8. Launch the visualisation
python src/app_demo.py
```

### Using the Application

Once the app is live at `http://127.0.0.1:8050/`, the Explorer page supports four interaction modes.

Filter by event. Use the event dropdown to isolate one of the seven disaster categories. The scatter plot greys out non matching points and the gallery shows ten random samples from the selected event.

Semantic text search. Type something like "flooded street with a submerged car" and press Enter or the Search button. Matches appear as blue dots on the scatter and the top ten show up as cards in the gallery.

Multimodal search. Click the upload box next to the search bar and drop in an image. You can also add a text refinement. Press Search. The image and text embeddings get blended with a weighted average and matches come back ranked by cosine similarity.

Visual query (image to image). Click any point on the scatter plot. The clicked image's embedding is compared against every other image using cosine similarity, and the nearest neighbours come back. Press Clear or type a new text query to reset.

### Running the Unit Tests

The five unit tests from Table 5 of the Final Report can be run with:

```bash
pytest tests/test_unit_report.py -v
```

All five assertions (L2 normalisation magnitude, sigmoid severity bounds, corrupted image rejection, UMAP NaN handling, and HDBSCAN noise assignment) should pass.

## Acknowledgements

This project was developed as part of an Honours dissertation under the supervision of XinHui Ma. The CrisisMMD dataset was provided by the Qatar Computing Research Institute.

## References

Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
Alam, F. et al. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. ICWSM.
Campello, R. J. G. B., Moulavi, D., and Sander, J. (2013). Density Based Clustering Based on Hierarchical Density Estimates. PAKDD.
Abnar, S. and Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL.
Edelsbrunner, H. and Harer, J. (2010). Computational Topology: An Introduction. AMS.
Ollivier, Y. (2009). Ricci Curvature of Markov Chains on Metric Spaces. Journal of Functional Analysis.
Zhai, X. et al. (2023). Sigmoid Loss for Language Image Pre Training. ICCV.
He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
