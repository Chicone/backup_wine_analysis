Overview
========

In this section, we address the classification of press wine samples. The
goal is to predict wine press class membership (e.g., A, B, or C) based on GC-MS data collected from Merlot and
Cabernet Sauvignon wines across multiple vintages.

Special care is taken to ensure that replicate samples are kept together during cross-validation, so that no
replicate of the same sample appears in both training and test sets. This avoids inflated performance
estimates due to duplicates present in train and trest sets and provides a more realistic measure of generalization.