"""Central project configuration file."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List
import torch


@dataclass
class Config:
    """Main configuration class for the project."""
    
    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    VISUALISATION_DIR: Path = DATA_DIR / "visualisation"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"

    # --- Image Processing ---
    IMAGE_SIZE: Tuple[int, int] = (224, 224)  # For CLIP and ResNet
    THUMBNAIL_SIZE: Tuple[int, int] = (128, 128)  # For visualisation sprites
    IMAGE_EXTENSIONS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
    MAX_IMAGE_SIZE_MB: float = 10.0
    
    # CLIP normalization constants
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    NORMALIZE_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    
    # --- Model Settings ---
    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_EMBEDDING_DIM: int = 512
    
    RESNET_MODEL_NAME: str = "resnet50"
    RESNET_EMBEDDING_DIM: int = 2048
    
    DEFAULT_MODEL: str = "clip"  # "clip" or "resnet"

    # --- Inference Settings ---
    BATCH_SIZE: int = 32  # Adjust based on GPU memory
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 2
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_MIXED_PRECISION: bool = True  # FP16 for faster inference

    # --- Dimensionality Reduction ---
    UMAP_N_NEIGHBORS: int = 15  # Controls local vs global structure
    UMAP_MIN_DIST: float = 0.1  # Minimum distance between points
    UMAP_METRIC: str = "cosine"
    UMAP_N_COMPONENTS: int = 2
    UMAP_RANDOM_STATE: int = 42

    TSNE_PERPLEXITY: float = 30.0
    TSNE_LEARNING_RATE: float = 200.0
    TSNE_N_ITER: int = 1000
    TSNE_RANDOM_STATE: int = 42
    
    PCA_N_COMPONENTS: int = 2
    PCA_WHITEN: bool = False
    
    # --- Clustering ---
    HDBSCAN_MIN_CLUSTER_SIZE: int = 10
    HDBSCAN_MIN_SAMPLES: int = 5
    HDBSCAN_METRIC: str = "euclidean"
    HDBSCAN_CLUSTER_SELECTION_METHOD: str = "eom"  # 'eom' or 'leaf'

    # --- Data Storage ---
    HDF5_COMPRESSION: str = "gzip"
    HDF5_COMPRESSION_LEVEL: int = 4  # Range: 0-9
    HDF5_CHUNK_SIZE: int = 100
    
    # --- Visualisation ---
    PLOT_WIDTH: int = 1200
    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"
    POINT_SIZE: int = 8
    POINT_OPACITY: float = 0.7
    
    DASH_HOST: str = "127.0.0.1"
    DASH_PORT: int = 8050
    DASH_DEBUG: bool = True
    
    # --- Dataset ---
    DATASET_NAME: str = "CrisisMMD"
    DATASET_SUBSET_SIZE: int = 3000  # Use subset for development (None for full)
    
    LABEL_COLUMNS: List[str] = field(default_factory=lambda: [
        'event_name', 'disaster_type', 'damage_severity', 'informativeness'
    ])
    
    # --- Evaluation ---
    EVALUATION_METRICS: List[str] = field(default_factory=lambda: [
        'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score',
        'trustworthiness', 'continuity'
    ])
    
    # --- Logging ---
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "% (asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = PROJECT_ROOT / "disaster_viz.log"
    
    # --- Miscellaneous ---
    RANDOM_SEED: int = 42
    TQDM_DISABLE: bool = False
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        dirs = [
            self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR, self.METADATA_DIR, self.VISUALISATION_DIR,
            self.REPORTS_DIR, self.FIGURES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_embedding_path(self, model_name: str) -> Path:
        """Get path for storing model embeddings."""
        return self.EMBEDDINGS_DIR / f"{model_name}_embeddings.h5"
    
    def get_reduction_path(self, model_name: str, method: str) -> Path:
        """Get path for storing reduced coordinates."""
        return self.VISUALISATION_DIR / f"{model_name}_{method}_coords.npy"
    
    def get_cluster_path(self, model_name: str, method: str) -> Path:
        """Get path for storing cluster labels."""
        return self.VISUALISATION_DIR / f"{model_name}_{method}_clusters.npy"
    
    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        summary = [
            "="*60, "DISASTER VISUALISATION CONFIGURATION", "="*60,
            f"Device: {self.DEVICE}", f"Default Model: {self.DEFAULT_MODEL}",
            f"Batch Size: {self.BATCH_SIZE}", f"Image Size: {self.IMAGE_SIZE}",
            f"Dataset: {self.DATASET_NAME}", f"Subset Size: {self.DATASET_SUBSET_SIZE}",
            "-"*60, "UMAP Settings:", f"  n_neighbors: {self.UMAP_N_NEIGHBORS}",
            f"  min_dist: {self.UMAP_MIN_DIST}", f"  metric: {self.UMAP_METRIC}",
            "-"*60, "HDBSCAN Settings:", f"  min_cluster_size: {self.HDBSCAN_MIN_CLUSTER_SIZE}",
            f"  min_samples: {self.HDBSCAN_MIN_SAMPLES}", "="*60,
        ]
        return "\n".join(summary)

# Global config instance
config = Config()

if __name__ == "__main__":
    print(config.summary())