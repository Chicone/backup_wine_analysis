# Wine Analysis Library Documentation

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
git clone https://github.com/pougetlab/wine_analysis.git
cd wine_analysis
git checkout wine-analysis-package
pip install -e .
pip install -r requirements.txt
```


## Overview of the cross-dataset classification pipeline 

The script in `main.py` contains instructions to calculate, among other things, the cross-dataset classification accuracy. 
This consists on training a classification model on one dataset (2018 oak) and testing it on another dataset (2022 oak). 
To ensure accurate comparison and classification, the chromatograms from both datasets are aligned or synchronised to a
common reference using two main algorithms, which are applied sequentially. The reference chromatogram is set to be the
mean chromatogram of the 2018 dataset.
_
### Step 1: Initial Alignment with `lag_profile_from_peaks()`

The first algorithm, `lag_profile_from_peaks()`, is responsible for calculating an initial alignment between target
 chromatograms and the reference. It does so by identifying peaks in both and trying to match them based on their 
distance, scaling sections accordingly. A peak is defined as any datapoint whose value is larger than the immediate 
datapoints to its left and right.
  
#### Global Alignment: 
There is an initial step where the target chromatograms are shifted globally in order maximize the cross-correlation 
with the reference chromatogram, providing a first rough alignment.  However, only the first third of the chromatograms 
is used in the computation of the cross-correlation. There seems to be non-linear relationships between peaks locations 
when comparing target and  reference chromatograms, and therefore  we focus on the first third to make sure 
that at least this part is correctly aligned. This is important because it provides a starting point for the 
next alignment steps. A Gaussian filter is applied to smooth both chromatograms before the global alignment. 

#### Between-peak Section Alignment: 
It can be summarized in the following points:
- After the global shift, both reference and target chromatograms are divided into segments and the location of the 
largest peak from each segment selected. Alignment then proceeds by matching selected peaks in the target chromatogram 
 with those in the reference. The matching is based on finding the closest peak in the reference to the current peak 
 in the target. If the closest peak is farther than an alignment tolerance of 40 retention time units, then we skip this 
 alignment  and move into the next peak in the target and so on. 
- Looking at each pair of matched peaks, the difference between their locations is used to calculate a scaling factor 
that is locally applied to the target chromatogram. The locality means that we only scale the interval between  
the current target peak and the (corrected) target peak from the previous segment, which was adjusted    
to the same location as the previous reference peak. 
- The output from this algorithm is a set of lag values that describe how much each segment of the target chromatogram 
 should be shifted to align with the reference.
- Using this lag profile, we can fit a spline and adjust the retention times of the original target chromatogram to 
  the reference in a very smooth fashion. The spline is allowed some flexibility to fit the points so that the 
  adjustment of peak locations is somehow soft.  
- This algorithm has the disadvantage that all the data between the peaks used for alignment are also scaled. 
  The problem with this is that if some peaks have location somehow incorrect or if they are incorrectly aligned, the 
 errors will also be transmitted to other peaks, potentially distorting the data too much. This is also the reason why 
this algorithm is only used as a preprocessing step to locally and roughly align the chromatograms based on the major peaks.
- As a stand-alone algorithm after the global alignment, it produces a cross-dataset classification accuracy of 65.3%.

### Step 2: Fine-Tuning with `lag_profile_moving_peaks_individually()`

After applying the initial global and between-peak alignment, the synchronization is refined using the algorithm in 
`lag_profile_moving_peaks_individually()`.
 - The function is designed to refine the alignment between the reference chromatogram and the target chromatogram by 
    adjusting the positions of individual peaks in the target chromatogram without affecting other peaks. 
 - Contrary to the previous algorithm, only the regions between the peaks and their closest neighbour peaks (as 
    opposed to the region between the peak in the peak in the previous segment) are scaled. 
 - The scaling is carried out on both sides of the peak. The left part is used to bring the peak to the location of the 
    reference whereas the right region is scaled in the opposite direction to compensate for the left shift, so that the 
  rest of the chromatogram stays largely unaffected.
 - The matching of target peaks to reference peaks is based on checking not only the closest reference peak, but 
 also its surrounding peaks. 
 - For each surrounding peak, the scaling is performed as if it was the actual matched peak, and the resulting segment 
   is compared with the reference segment to see how well they match. The surrounding peak that matches the best is
   selected for the scaling.
 - The comparison is based on checking, after the scaling, the average separation between target and reference peaks 
   not only within the segment, but also within a significant portion of signal after it (e.g. 3000 retention time units). 
   This makes the comparison more robust.
 - A disadvantage of this algorithm is that it is very sensitive to changes in hyperparameters and the latter must be
   tuned very carefully.
 - Applied after the global and between-peak alignments, this algorithm produces a cross-dataset classification accuracy 
   of 84.0%.

### Final Step: Classification and Evaluation

With the chromatograms from both datasets now synchronized to a common reference, the script proceeds to train a 
classification model using the synchronized chromatograms from the 2018 oak dataset. This model is then tested on the 
synchronized chromatograms from the 2022 oak dataset to evaluate cross-dataset accuracy.

- **Model Training**:
  - The classifier (LDA) is trained on the synchronized chromatograms from the 2018 oak dataset. The features used for 
  classification are the intensities at specific retention times.

- **Model Testing**:
  - The trained classifier is then applied to the synchronized chromatograms from the 2022 oak dataset to predict the 
  labels.

- **Evaluation**:
  - The script calculates the accuracy of the model by comparing the predicted labels to the actual labels in the 2022 
  oak dataset. This accuracy indicates how well the model generalizes from one dataset to another, reflecting the 
  robustness of the synchronization and classification process.

This workflow ensures that the features used for classification are consistent across datasets, leading to more reliable
and generalizable models when analyzing chromatograms from different sources.
