"""
Central Configuration File
All project settings, paths, and hyperparameters in one place.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List
import torch


@dataclass
class Config:
    """Main configuration class for the disaster visualisation project."""
    

    # PATHS

    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    VISUALISATION_DIR: Path = DATA_DIR / "visualisation"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"


    # IMAGE PROCESSING

    IMAGE_SIZE: Tuple[int, int] = (224, 224)  # Standard for CLIP and ResNet
    THUMBNAIL_SIZE: Tuple[int, int] = (128, 128)  # For visualisation
    IMAGE_EXTENSIONS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
    MAX_IMAGE_SIZE_MB: float = 10.0  # Skip images larger than this
    
    # Image preprocessing
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)  # CLIP normalization
    NORMALIZE_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    

    # MODEL SETTINGS

    # CLIP settings
    CLIP_MODEL_NAME: str = "ViT-B/32"  # Options: ViT-B/32, ViT-B/16, ViT-L/14
    CLIP_EMBEDDING_DIM: int = 512  # ViT-B/32 and ViT-B/16 output 512-dim
    
    # ResNet settings
    RESNET_MODEL_NAME: str = "resnet50"  # Options: resnet50, resnet101, resnet152
    RESNET_EMBEDDING_DIM: int = 2048  # ResNet50/101/152 final layer
    
    # Model selection
    DEFAULT_MODEL: str = "clip"  # Options: "clip", "resnet50"

    # TRAINING/INFERENCE SETTINGS

    BATCH_SIZE: int = 32  # Adjust based on GPU memory (RTX 2070: 16-32, RTX 3090: 64-128)
    NUM_WORKERS: int = 4  # DataLoader workers
    PIN_MEMORY: bool = True  # Faster GPU transfer
    PREFETCH_FACTOR: int = 2  # Pre-load batches
    
    # GPU settings

    if torch.cuda.is_available():
        DEVICE: str = "cuda"
    else:
        DEVICE: str = "cpu"
        
    USE_MIXED_PRECISION: bool = True  # FP16 for faster inference
    

    # DIMENSIONALITY REDUCTION
    
    # UMAP parameters
    UMAP_N_NEIGHBORS: int = 15  # Controls local vs global structure (5-50)
    UMAP_MIN_DIST: float = 0.1  # Minimum distance between points (0.0-0.99)
    UMAP_METRIC: str = "cosine"  # Distance metric
    UMAP_N_COMPONENTS: int = 2  # Output dimensions (2 for visualisation)
    UMAP_RANDOM_STATE: int = 42  # For reproducibility
    
    # t-SNE parameters
    TSNE_PERPLEXITY: float = 30.0  # Balance local vs global (5-50)
    TSNE_LEARNING_RATE: float = 200.0  # Step size
    TSNE_N_ITER: int = 1000  # Number of iterations
    TSNE_RANDOM_STATE: int = 42
    
    # PCA parameters
    PCA_N_COMPONENTS: int = 2  # Output dimensions
    PCA_WHITEN: bool = False  # Normalize components
    

    # CLUSTERING

    # HDBSCAN parameters
    HDBSCAN_MIN_CLUSTER_SIZE: int = 10  # Minimum points to form a cluster
    HDBSCAN_MIN_SAMPLES: int = 5  # Core points threshold
    HDBSCAN_METRIC: str = "euclidean"  # Distance metric for clustering
    HDBSCAN_CLUSTER_SELECTION_METHOD: str = "eom"  # 'eom' or 'leaf'

    # DATA STORAGE

    # HDF5 settings
    HDF5_COMPRESSION: str = "gzip"  # Compression algorithm
    HDF5_COMPRESSION_LEVEL: int = 4  # 0-9, higher = more compression
    HDF5_CHUNK_SIZE: int = 100  # Chunk size for HDF5 storage
    

    # VISUALISATION

    # Plotly settings
    PLOT_WIDTH: int = 1200
    PLOT_HEIGHT: int = 800
    PLOT_TEMPLATE: str = "plotly_dark"  # Theme
    POINT_SIZE: int = 8
    POINT_OPACITY: float = 0.7
    
    # Dash app settings
    DASH_HOST: str = "127.0.0.1"
    DASH_PORT: int = 8050
    DASH_DEBUG: bool = True
    
    # ============================================================================
    # DATASET SPECIFIC
    # ============================================================================
    # CrisisMMD dataset
    DATASET_NAME: str = "CrisisMMD"
    DATASET_SUBSET_SIZE: int = 3000  # Use subset for development (None for full)
    
    # Label columns
    LABEL_COLUMNS: List[str] = field(default_factory=lambda: [
        'event_name',
        'disaster_type', 
        'damage_severity',
        'informativeness'
    ])
    

    # EVALUATION

    EVALUATION_METRICS: List[str] = field(default_factory=lambda: [
        'silhouette_score',
        'davies_bouldin_score',
        'calinski_harabasz_score',
        'trustworthiness',
        'continuity'
    ])
    

    # LOGGING

    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = PROJECT_ROOT / "disaster_viz.log"
    


    RANDOM_SEED: int = 42  # Global random seed for reproducibility
    TQDM_DISABLE: bool = False  # Disable progress bars if needed
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.METADATA_DIR,
            self.VISUALISATION_DIR,
            self.REPORTS_DIR,
            self.FIGURES_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_embedding_path(self, model_name: str) -> Path:
        """Get path for storing embeddings for a specific model."""
        return self.EMBEDDINGS_DIR / f"{model_name}_embeddings.h5"
    
    def get_reduction_path(self, model_name: str, method: str) -> Path:
        """Get path for storing reduced coordinates."""
        return self.VISUALISATION_DIR / f"{model_name}_{method}_coords.npy"
    
    def get_cluster_path(self, model_name: str, method: str) -> Path:
        """Get path for storing cluster labels."""
        return self.VISUALISATION_DIR / f"{model_name}_{method}_clusters.npy"
    
    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        summary = []
        summary.append("="*60)
        summary.append("DISASTER VISUALISATION CONFIGURATION")
        summary.append("="*60)
        summary.append(f"Device: {self.DEVICE}")
        summary.append(f"Default Model: {self.DEFAULT_MODEL}")
        summary.append(f"Batch Size: {self.BATCH_SIZE}")
        summary.append(f"Image Size: {self.IMAGE_SIZE}")
        summary.append(f"Dataset: {self.DATASET_NAME}")
        summary.append(f"Subset Size: {self.DATASET_SUBSET_SIZE}")
        summary.append("-"*60)
        summary.append("UMAP Settings:")
        summary.append(f"  n_neighbors: {self.UMAP_N_NEIGHBORS}")
        summary.append(f"  min_dist: {self.UMAP_MIN_DIST}")
        summary.append(f"  metric: {self.UMAP_METRIC}")
        summary.append("-"*60)
        summary.append("HDBSCAN Settings:")
        summary.append(f"  min_cluster_size: {self.HDBSCAN_MIN_CLUSTER_SIZE}")
        summary.append(f"  min_samples: {self.HDBSCAN_MIN_SAMPLES}")
        summary.append("="*60)
        return "\n".join(summary)


# Create a global config instance
config = Config()


if __name__ == "__main__":
    # Test the configuration
    print(config.summary())
    print(f"\nProject root: {config.PROJECT_ROOT}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")