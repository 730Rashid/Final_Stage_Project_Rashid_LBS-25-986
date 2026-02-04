"""
Central Project Configuration.

This module contains all the configuration settings for the project.
Using a centralised configuration makes it easy to adjust parameters
and ensures consistency across all modules.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List
import torch


@dataclass
class Config:
    """
    Main configuration class for the project.
    
    All settings are defined as class attributes with sensible defaults.
    The settings are optimised for my hardware (MX450 GPU, 16GB RAM).
    """
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    VISUALISATION_DIR: Path = DATA_DIR / "visualisation"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"

    # Image Processing
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    THUMBNAIL_SIZE: Tuple[int, int] = (128, 128)
    IMAGE_EXTENSIONS: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"]
    )
    MAX_IMAGE_SIZE_MB: float = 10.0
    
    # CLIP normalisation constants
    NORMALISE_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    NORMALISE_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    
    # Model Settings
    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_EMBEDDING_DIM: int = 512
    
    RESNET_MODEL_NAME: str = "resnet50"
    RESNET_EMBEDDING_DIM: int = 2048
    
    DEFAULT_MODEL: str = "clip"

    # Inference Settings
    # Batch size is set low for MX450 with 2GB VRAM
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 2
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_MIXED_PRECISION: bool = True

    # Dimensionality Reduction (UMAP)
    UMAP_N_NEIGHBOURS: int = 15
    UMAP_MIN_DIST: float = 0.1
    UMAP_METRIC: str = "cosine"
    UMAP_N_COMPONENTS: int = 2
    UMAP_RANDOM_STATE: int = 42

    # Dimensionality Reduction (t-SNE)
    TSNE_PERPLEXITY: float = 30.0
    TSNE_LEARNING_RATE: float = 200.0
    TSNE_N_ITER: int = 1000
    TSNE_RANDOM_STATE: int = 42
    
    # Dimensionality Reduction (PCA)
    PCA_N_COMPONENTS: int = 2
    PCA_WHITEN: bool = False
    
    # Clustering (HDBSCAN)
    HDBSCAN_MIN_CLUSTER_SIZE: int = 10
    HDBSCAN_MIN_SAMPLES: int = 5
    HDBSCAN_METRIC: str = "euclidean"
    HDBSCAN_CLUSTER_SELECTION_METHOD: str = "eom"

    # Data Storage (HDF5)
    HDF5_COMPRESSION: str = "gzip"
    HDF5_COMPRESSION_LEVEL: int = 4
    HDF5_CHUNK_SIZE: int = 100
    
    # Visualisation
    PLOT_WIDTH: int = 1200
    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"
    POINT_SIZE: int = 8
    POINT_OPACITY: float = 0.7
    
    # Dash Server
    DASH_HOST: str = "127.0.0.1"
    DASH_PORT: int = 8050
    DASH_DEBUG: bool = True
    
    # Dataset
    DATASET_NAME: str = "CrisisMMD"
    DATASET_SUBSET_SIZE: int = 3000
    
    LABEL_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "event_name", "disaster_type", "damage_severity", "informativeness"
        ]
    )
    
    # Analytics
    ANALYTICS_SAMPLE_SIZE: int = 500  # pairwise sim sampling per event

    # Search & Classification Thresholds
    SEARCH_MIN_THRESHOLD: float = 0.28
    CLASSIFICATION_THRESHOLD: float = 0.22
    SEARCH_MIN_QUERY_LENGTH: int = 3
    SEARCH_MAX_QUERY_LENGTH: int = 200

    # Privacy (Face Blurring)
    FACE_BLUR_KERNEL: Tuple[int, int] = (99, 99)
    FACE_BLUR_SIGMA: int = 30
    FACE_BLUR_PADDING: float = 0.1
    FACE_DETECT_SCALE_FACTOR: float = 1.05
    FACE_DETECT_MIN_NEIGHBORS: int = 3
    FACE_DETECT_MIN_SIZE: Tuple[int, int] = (20, 20)

    # Vectorisation
    VECTORISE_BATCH_SIZE: int = 32

    # Evaluation
    EVALUATION_METRICS: List[str] = field(
        default_factory=lambda: [
            "silhouette_score", 
            "davies_bouldin_score", 
            "calinski_harabasz_score",
            "trustworthiness", 
            "continuity"
        ]
    )
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = PROJECT_ROOT / "disaster_viz.log"
    
    # Miscellaneous
    RANDOM_SEED: int = 42
    TQDM_DISABLE: bool = False
    
    def __post_init__(self):
        """Create necessary directories after initialisation."""
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create all necessary directories if they do not exist."""
        dirs = [
            self.DATA_DIR, 
            self.RAW_DATA_DIR, 
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR, 
            self.METADATA_DIR, 
            self.VISUALISATION_DIR,
            self.REPORTS_DIR, 
            self.FIGURES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_embedding_path(self, model_name: str) -> Path:
        """Get path for storing model embeddings."""
        return self.EMBEDDINGS_DIR / "{}_embeddings.h5".format(model_name)
    
    def get_reduction_path(self, model_name: str, method: str) -> Path:
        """Get path for storing reduced coordinates."""
        return self.VISUALISATION_DIR / "{}_{}_coords.npy".format(model_name, method)
    
    def get_cluster_path(self, model_name: str, method: str) -> Path:
        """Get path for storing cluster labels."""
        return self.VISUALISATION_DIR / "{}_{}_clusters.npy".format(model_name, method)
    
    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        lines = [
            "DISASTER VISUALISATION CONFIGURATION",
            "Device: {}".format(self.DEVICE),
            "Default Model: {}".format(self.DEFAULT_MODEL),
            "Batch Size: {}".format(self.BATCH_SIZE),
            "Image Size: {}".format(self.IMAGE_SIZE),
            "Dataset: {}".format(self.DATASET_NAME),
            "Subset Size: {}".format(self.DATASET_SUBSET_SIZE),
            "UMAP Settings:",
            "  n_neighbours: {}".format(self.UMAP_N_NEIGHBOURS),
            "  min_dist: {}".format(self.UMAP_MIN_DIST),
            "  metric: {}".format(self.UMAP_METRIC),
            "HDBSCAN Settings:",
            "  min_cluster_size: {}".format(self.HDBSCAN_MIN_CLUSTER_SIZE),
            "  min_samples: {}".format(self.HDBSCAN_MIN_SAMPLES),
        ]
        return "\n".join(lines)


# Global config instance
config = Config()


if __name__ == "__main__":
    print(config.summary())