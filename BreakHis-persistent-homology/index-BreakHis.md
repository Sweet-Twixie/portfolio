---
layout: default
title: Persistent homology on Breast Cancer histopathology dataset
---

# About the project 

This project applies **persistent homology** to the **BreakHis breast cancer histopathology dataset** to perform **binary classification (benign vs. malignant)**. The workflow combines stain normalisation, cubical persistent homology (via Cripser), topological feature vectorisation (persistence statistics and persistence images), and classical machine-learning models such as Random Forests and XGBoost. The goal is to demonstrate how topological data analysis (TDA) can capture meaningful global tissue structure and outperform standard handcrafted texture descriptors on this task.


# Highlights
- **Stain normalization** with `staintools` (reduces color variation).
- **Data augmentation**: random rotation, flips, zoom, brightness jitter.
- **TDA features**:
  - **Cubical homology** (`cripser`) on grayscale → persistence diagrams.
  - **Persistence Images** (Persim) **or** **persistent summary statistics**.
- **Classifiers**: `RandomForestClassifier` and `XGBClassifier` (XGBoost).
- **Performance**: ~90% accuracy (varies by magnification/split).

## 🔍 Explore Sections
- [About the dataset](breakhis.html) – provides details about the BreakHis dataset and describes the preprocessing workflow.
- [Persistent Homology](persistent-homology.html) – covers the computation of persistence diagrams using cubical homology and the vectorisation methods (persistence statistics and persistence images).


# Results 

RF - Random Forest 
XGB - XGBoost
stats - Persistence Statistics 
PI - Persistence Images

| Magnification | RF on stats | XGB on stats | RF on PI (H₀) | RF on PI (H₁) |
|--------------:|-----------:|-------------:|--------------:|--------------:|
| 40×           | 0.89       | 0.90         | 0.85          | 0.78          |
| 100×          | 0.88       | 0.88         | 0.84          | 0.72          |
| 200×          | 0.80       | 0.83         | 0.80          | 0.75          |
| 400×          | 0.80       | 0.82         | 0.78          | 0.73          |

# Discussion 
Across magnifications, persistence statistics performed as well as — or better than — persistence images (PI), especially for the H₁ features where topological structure is weaker. We found that expanding the statistical feature set beyond a compact group (count, total persistence, and basic lifetime statistics) actually reduced accuracy, suggesting that additional descriptors introduced noise or redundancy. Similarly, combining PI features from H₀ and H₁ was less effective than using H₀ alone, indicating that the H₁ information did not generalise well under our settings.

For comparison, standard handcrafted texture features such as grey-level co-occurrence matrices (GLCM), local binary patterns (LBP), and local phase quantisation (LPQ) achieved roughly 0.74 accuracy with a Random Forest using the same stain normalisation and augmentation pipeline - noticeably below the performance of the topological approaches. These results suggest that persistent homology, which captures global structural organisation, provides a stronger and more discriminative signal than classical local texture descriptors for this task.

Further improvements could arise from tuning PI hyperparameters (resolution, kernel bandwidth, weighting), expanding and then selecting informative statistical features, or combining global topological descriptors with local features (GLCM/LBP/LPQ) or deep embeddings to leverage complementary information.



# 🔗 Repository
[View the code on GitHub]([https://github.com/Sweet-Twixie/football-classification](https://github.com/Sweet-Twixie/Spectral-Sequences-and-Topological-Data-Analysis-for-Image-Processing))

