> 📦 **Note:** This documentation refers to the `wine-analysis-package` branch, which contains the most accessible and minimal version of the GC-MS Wine Analysis tools. 
> It is intended for testing and basic usage.  
> Other branches may contain experimental or extended versions

# General Documentation

Welcome to the **Wine Analysis Library** documentation!

## Overview

The Wine Analysis Library is a comprehensive toolkit designed for analyzing and processing wine-related data. 
The library provides various modules to facilitate data loading, preprocessing, dimensionality reduction, 
classification, and visualization of wine chromatograms and related datasets.

### Key Features

- **Data Loading & Preprocessing**: Load and preprocess wine datasets efficiently using custom utilities.
- **Dimensionality Reduction**: Apply various dimensionality reduction techniques like PCA (Principal Component Analysis) to simplify complex datasets.
- **Classification**: Use machine learning classifiers to categorize wine samples based on their chemical compositions or other features.
- **Visualization**: Generate informative visualizations, including chromatograms and scatter plots, to explore and present the data effectively.
- **Analysis**: Perform detailed analysis on wine data, including peak detection and alignment across samples.

## Installation

This repository contains multiple development branches for different use cases and experimental pipelines.
The wine-analysis-package branch is the simplest and most stable version, specifically intended for basic GC-MS data analysis workflows. It includes the core functionalities for chromatogram preprocessing, alignment, classification, and visualization, and is ideal for most users working with wine or chemical analysis datasets.

To use this version, make sure to clone and switch to this branch:
```bash
# Clone the repository and switch to the correct branch
git clone https://github.com/pougetlab/wine_analysis.git
cd wine_analysis
git checkout wine-analysis-package

# (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

Some modules in this library may require extra dependencies that are not automatically listed in requirements.txt. 
If you encounter import errors when running scripts, make sure to install the following commonly used packages:
```bash
pip install torch torchvision pynndescent netCDF4 seaborn umap-learn tqdm scikit-optimize
```

## Preparing the GC-MS Data

Before running the analysis scripts, your GC-MS data must be prepared in a specific directory structure.

### Required Format

Each sample must be stored in its own `.D` folder (as typically exported by Agilent ChemStation or similar software). 
For example:
```
datasets/
├── PINOT_NOIR/
│ ├── Sample1.D/
│ ├── Sample2.D/
│ └── ...
└──  ...
```
Then, within each sample there should be a CSV file like this:
![img.png](img.png)
, where the first column is the retention time and the next columns are the intensity signals of each m/z channel 
(starting at 40 in this example).

## Running Scripts
To execute one of the analysis scripts, navigate to the root of the project (where the scripts/ directory is located) and 
run the script using Python. For example, to run the Pinot Noir classification pipeline:
```bash
python scripts/pinot_noir/train_test_pinot_noir.py
```

Note: Each script is documented in detail in the corresponding section of the online documentation.