"""
Setup script for disaster-viz package.
Makes the package installable in development mode.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="disaster-visualisation",
    version="0.1.0",
    author="Your Name",
    author_email="R.A.Pandor-2022@hull.ac.uk",
    description="Interactive visualisation system for natural disaster image embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/730Rashid/Final_Stage_Project_Rashid_LBS-25-986",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualisation",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.3",
        "hdbscan>=0.8.33",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "h5py>=3.9.0",
        "plotly>=5.15.0",
        "dash>=2.11.0",
        "streamlit>=1.25.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "disaster-viz=scripts.run_visualisation:main",
        ],
    },
)