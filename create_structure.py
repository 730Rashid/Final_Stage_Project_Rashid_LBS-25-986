"""
Complete Project Structure Setup Script
Creates the entire disaster-visualisation project structure with all files and folders.
Run this after creating your virtual environment and activating it.

Usage:
    python create_structure.py
"""

import os
from pathlib import Path


def create_file(path: Path, content: str = ""):
    """Create a file with optional content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"    Created: {path}")


def create_project_structure():
    """Create the complete project directory structure."""
    
    print("="*70)
    print("DISASTER-VISUALISATION PROJECT STRUCTURE SETUP")
    print("="*70)
    print()
    
    base = Path.cwd()
    
    # Define all directories
    directories = [
        'config',
        'data/raw',
        'data/processed',
        'data/embeddings',
        'data/metadata',
        'data/visualisation',
        'src',
        'src/preprocessing',
        'src/embeddings',
        'src/reduction',
        'src/clustering',
        'src/visualisation',
        'src/evaluation',
        'src/utils',
        'notebooks',
        'scripts',
        'tests',
        'reports/figures/model_comparison',
        'reports/figures/reduction_comparison',
        'reports/figures/cluster_evaluation',
        'reports/metrics',
    ]
    
    print("Creating directories...")
    for directory in directories:
        dir_path = base / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  {directory}/")
    
    print("\nCreating __init__.py files...")
    init_dirs = [
        'config',
        'src',
        'src/preprocessing',
        'src/embeddings',
        'src/reduction',
        'src/clustering',
        'src/visualisation',
        'src/evaluation',
        'src/utils',
        'tests',
    ]
    
    for directory in init_dirs:
        init_file = base / directory / '__init__.py'
        create_file(init_file, '')
    
    print("\nCreating .gitkeep files...")
    gitkeep_dirs = [
        'data/raw',
        'data/processed',
        'data/embeddings',
        'data/metadata',
        'data/visualisation',
        'reports/figures/model_comparison',
        'reports/figures/reduction_comparison',
        'reports/figures/cluster_evaluation',
        'reports/metrics',
    ]
    
    for directory in gitkeep_dirs:
        gitkeep = base / directory / '.gitkeep'
        create_file(gitkeep, '')
    
    print("\nCreating placeholder Python files...")
    
    # Preprocessing modules
    preprocessing_files = [
        'src/preprocessing/image_loader.py',
        'src/preprocessing/preprocessor.py',
        'src/preprocessing/duplicate_detector.py',
        'src/preprocessing/metadata_extractor.py',
    ]
    
    # Embeddings modules
    embeddings_files = [
        'src/embeddings/clip_encoder.py',
        'src/embeddings/resnet_encoder.py',
        'src/embeddings/encoder_factory.py',
        'src/embeddings/batch_processor.py',
        'src/embeddings/embedding_storage.py',
    ]
    
    # Reduction modules
    reduction_files = [
        'src/reduction/reducer_base.py',
        'src/reduction/umap_reducer.py',
        'src/reduction/tsne_reducer.py',
        'src/reduction/pca_reducer.py',
    ]
    
    # Clustering modules
    clustering_files = [
        'src/clustering/hdbscan_clusterer.py',
        'src/clustering/cluster_analyzer.py',
    ]
    
    # Visualisation modules
    visualisation_files = [
        'src/visualisation/dash_app.py',
        'src/visualisation/plotly_components.py',
        'src/visualisation/image_server.py',
        'src/visualisation/evaluation_panel.py',
        'src/visualisation/comparison_view.py',
    ]
    
    # Evaluation modules
    evaluation_files = [
        'src/evaluation/metrics.py',
        'src/evaluation/cluster_quality.py',
        'src/evaluation/comparison_utils.py',
    ]
    
    # Utility modules  
    utils_files = [
        'src/utils/metrics.py',
    ]
    
    # Scripts
    script_files = [
        'scripts/download_dataset.py',
        'scripts/preprocess_images.py',
        'scripts/generate_embeddings.py',
        'scripts/reduce_dimensions.py',
        'scripts/cluster_embeddings.py',
        'scripts/compare_models.py',
        'scripts/compare_reductions.py',
        'scripts/run_visualisation.py',
    ]
    
    # Tests
    test_files = [
        'tests/test_preprocessing.py',
        'tests/test_embeddings.py',
        'tests/test_reduction.py',
        'tests/test_clustering.py',
    ]
    
    # Notebooks
    notebook_files = [
        'notebooks/01_data_exploration.ipynb',
        'notebooks/02_embedding_experiments.ipynb',
        'notebooks/03_reduction_comparison.ipynb',
        'notebooks/04_clustering_analysis.ipynb',
        'notebooks/05_model_comparison.ipynb',
        'notebooks/06_reduction_comparison.ipynb',
        'notebooks/07_cluster_evaluation.ipynb',
    ]
    
    all_files = (
        preprocessing_files + embeddings_files + reduction_files +
        clustering_files + visualisation_files + evaluation_files +
        utils_files + script_files + test_files
    )
    
    placeholder = '"""TODO: Implement this module."""\n'
    
    for file_path in all_files:
        create_file(base / file_path, placeholder)
    
    # Create empty notebook files
    print("\nCreating Jupyter notebooks...")
    notebook_template = '''{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    for notebook in notebook_files:
        create_file(base / notebook, notebook_template)
    
    print("\n" + "="*70)
    print(" Project structure created successfully! ")
    print("="*70)
    print("\nNext steps:")
    print("1. Copy the config files (settings.py, logging_config.py)")
    print("2. Copy the utility files (file_utils.py, gpu_utils.py)")
    print("3. Copy requirements.txt and setup.py")
    print("4. Copy verify_install.py and .gitignore")
    print("5. Create virtual environment: python -m venv venv")
    print("6. Activate venv: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux/Mac)")
    print("7. Install dependencies: pip install -r requirements.txt")
    print("8. Run verification: python verify_install.py")
    print("9. Initialise git: git init && git add . && git commit -m 'Initial structure'")
    print()


if __name__ == "__main__":
    try:
        create_project_structure()
    except Exception as e:
        print(f"\n Error: {e}")
        print("Make sure you're running this from an empty directory or")
        print("a directory where you want to create the project structure.")