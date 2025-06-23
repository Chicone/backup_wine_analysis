# Web Interface Documentation

> 📘 **Note:** This documentation refers to the `wine_analysis_web_interface` branch, which includes a dynamic frontend for GC-MS wine analysis. For core scripts and broader usage, refer to the main documentation at [https://pougetlab.github.io/wine_analysis/](https://pougetlab.github.io/wine_analysis/).

## Overview

This web-based tool allows interactive configuration and execution of GC-MS wine analysis tasks. It supports classification, sensory prediction, dimensionality reduction, and model interpretation, all from a graphical user interface.

### Key Features

- **Multiple Wine Families**: Bordeaux, Pinot Noir, Press Wines, and Champagne.
- **Dynamic Configuration**: Selecting different wine families and tasks renders different configuration options.
- **Classification Pipelines**: Train and evaluate classifiers for wine origin and vintage prediction.
- **Champagne Sensory Modeling**: Predict sensory attributes with chromatograms, compare tasters, visualize taster attention maps.
- **Dimensionality Reduction**: Plot PCA, UMAP or t-SNE projections in 2D or 3D.
- **Live Logs**: Real-time logs from backend operations.

---

## Installation

### 1. Clone the Repository and Use the Correct Branch
```bash
git clone https://github.com/pougetlab/wine_analysis.git
cd wine_analysis
git checkout wine_analysis_web_interface
```

### 2. Install Backend Requirements
See general documentation for more details.
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Start the Backend
```bash
cd api_web
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the Frontend
```bash
cd ../frontend
npm install
npm start
```
This should open the interface in your web browser.

---

## Dataset Preparation

Ensure your datasets follow the expected structure and naming. Update dataset paths in the code if needed.

Each sample should be a `.D` folder with an internal CSV containing retention time and m/z channel intensities. 
Metadata files (like CSVs with sample labels) must match the selected wine family and task. Champagne is a special case 
as only TICs (Total Ion Chromatograms) are available but thds structure of directories should be the same.

Wine families:
- **Bordeaux** → `bordeaux_oak`, `bordeaux_new`
- **Pinot Noir** → `pinot_regions`
- **Press Wines** → `press_all`, `press_grouped`
- **Champagne** → `heterocyc`, `aldehydes`, etc.

---

## Configuration via Interface

Navigate to the app in your browser (usually [http://localhost:3000](http://localhost:3000)) and select:

- **Wine Family**
- **Main Task** (Classification, Predict Age, etc.)
- **Subtask** if available (e.g., Shuffled Labels, Taster Scaling)
- **Model Type**: Ridge, LDA, Lasso Regression, etc.
- **Normalization**: Standard feature scaling.
- **Projection Options**: PCA/UMAP/t-SNE and source (e.g., classification scores or features).
- **Output Visuals**: Enable confusion matrices, attention heatmaps, logs, etc.

---

## Output and Visualizations

After execution, you'll see:

- Classification accuracy and confusion matrices
- Sensory attribute prediction scores
- 2D/3D plots from dimensionality reduction
- Per-taster model results (for Champagne)
- Live backend logs

Best is to look at the Logs for clean output as the  output console can be cluttered with information
There is also a documentation panel for quick reference of the interface usage.
---

## Need Help?
- For backend errors, check the terminal where `uvicorn` is running.
- For dataset issues, verify that paths and metadata match your task.

---

For full pipeline explanations and core script details, refer to the general documentation:
👉 [https://pougetlab.github.io/wine_analysis/](https://pougetlab.github.io/wine_analysis/)
