---
layout: default
title: Feature Selection
---

# Persistent Homology

Persistent homology has emerged as a powerful tool in biomedical image analysis, particularly for studying the structural and morphological properties of complex biological tissues. In cancer research, persistent homology has been applied to breast cancer histology for the analysis of nuclear architecture and tissue organization [2, 12, 8, 14], colorectal tumor segmentation based on glandular structures [12, 10], and
melanoma subtype classification from dermatoscopic images [4]. These applications demonstrate the ability of topological descriptors to capture multiscale features that are often invisible to classical pixel-based or texture-based methods, thus enhancing classification performance and offering novel insights into tumor biology.

Beyond histopathology, persistent homology has been incorporated into radiomics workflows for medical imaging modalities such as MRI and CT. For instance, persistent homology-based features have been utilized for the classification of liver tumors [11, 9], prognosis prediction in lung cancer using CT imaging [13, 7], and breast lesion detection in mammograms [3, 5, 15]. In these settings, topological features have been shown to complement standard radiomic descriptors, improving the discrimination of tumor types and patient outcomes.

Persistence diagrams provide a compact summary of the topological features present in an image across multiple intensity scales. Each point in a diagram represents the birth and death of a topological structure—such as connected components (H₀) or loops (H₁)—as the image is filtered. Importantly, persistence diagrams are **stable**: small perturbations or noise in the input image lead to only small changes in the diagram. This stability makes them reliable descriptors for downstream machine-learning tasks, enabling robust classification based on underlying tissue morphology.

---

# Persistent Cubical Homology on images

Persistent homology on images is typically computed using **cubical homology**, which is well-suited for data structured on a regular pixel grid. In this project, persistent homology was extracted from the grayscale (red-channel) images using **Cripser**, a fast and memory-efficient library for computing cubical complexes. Cripser operated directly on the preprocessed intensity values, producing persistence diagrams that capture the multiscale topological structure of the tissue images.

The code block below demonstrates how persistence diagrams are extracted from the grayscale images using **Cripser**, followed by filtering to remove points with infinite death time. These filtered diagrams serve as the foundation for subsequent vectorisation steps used in machine-learning classification.


```python
# Extracting Persistence Diagrams from the images

pds = []
for pc in tqdm(X_gray, desc="Computing persistence diagrams"):
    # compute persistence
    pd_diagram = cripser.computePH(pc, maxdim=1)
    
    # split into H0 and H1
    pd_diagram = [pd_diagram[pd_diagram[:, 0] == i, 1:3] for i in range(2)]

    # filter (keep only points with death < 1)
    pd_diagram_filtered = [
        pd_diagram[i][pd_diagram[i][:, 1] < 1]
        for i in range(2)
    ]

    pds.append(pd_diagram_filtered)

```

---

## 🔹 General Category Selection
To classify players into **Defender, Midfielder, Forwarder**, three complementary feature selection methods were applied:

1. **Random Forest Feature Importance** → Ranked features by importance.  
2. **Recursive Feature Elimination (RFE)** → Iteratively removed least important features until the optimal subset was found.  
3. **Gradient Boosting Feature Importance** → Validated feature relevance through boosting models.  

### ✅ Final Selected Features (for general classification)
After comparing the results, I finalized a set of **10 features** that consistently showed strong predictive power:

- `attacking_heading_accuracy_standardized`  
- `attacking_short_passing_standardized`  
- `skill_long_passing_standardized`  
- `power_strength_standardized`  
- `defending_average`  
- `defending_category_encoded`  
- `shooting_standardize`  
- `mentality_interceptions_categories_encoded`  
- `dribbling_standardize`  
- `mentality_positioning_categories_encoded`  

These features represent a balanced mix of **attacking, defending, physical, and mentality skills**, making them highly informative for separating Defenders, Midfielders, and Forwarders.

---

## 🔹 Category-Specific Feature Selection
In addition to general classification, I performed **feature selection separately for each category** (**Defender, Midfielder, Forwarder**) because each role requires different skill sets.  

For each subset, I applied **Random Forest Feature Importance** to select the **top 10 most relevant features** that best predict the specific positions within that category.

### Example: Forwarder Feature Selection
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X_forwarder = df_forwarder.drop(['ST', 'LW', 'RW', 'CF', 'LWB', 'RWB'], axis=1)
y_forwarder = df_forwarder[['ST', 'LW', 'RW', 'CF', 'LWB', 'RWB']]

y_forwarder_encoded = LabelEncoder().fit_transform(y_forwarder.values.argmax(axis=1))

X_train, X_test, y_train, y_test = train_test_split(X_forwarder, y_forwarder_encoded,
                                                    test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
rf_selected_features = rf_feature_importance.sort_values(ascending=False).head(10)

print("Random Forest Selected Features:\n", rf_selected_features)

```

# References

1. Adams, H., Chepushtanova, S., Emerson, T., Hanson, E., Kirby, M., Motta, F.,
   Neville, R., Peterson, C., Shipman, P., & Ziegelmeier, L. (2017).
   *Persistence Images: A Stable Vector Representation of Persistent Homology*.

2. Adcock, A., Carlsson, G., & Carlsson, E. (2016).
   *The ring of algebraic functions on persistence bar codes*. HHA, 18(1), 381–402.

3. Asaad, M. et al. (2022).
   *A topological machine learning framework for breast cancer detection using mammographic images*.
   Computers in Biology and Medicine, 142, 105217.

4. Chung, M. K. et al. (2018).
   *Topological data analysis for brain artery trees*. Annals of Applied Statistics, 12(2), 911–936.

5. de Silva, V., & Carlsson, G. (2004).
   *Topological estimation using witness complexes*. Eurographics Symposium on Point-Based Graphics, 157–166.

6. Edelsbrunner, H., & Harer, J. (2015).
   *Computational Topology: An Introduction*. arXiv:1507.06217.

7. González, G. et al. (2021).
   *A topological machine learning approach for non–small cell lung cancer prognostication using CT images*.
   Medical Image Analysis, 68, 101884.

8. Lawson, J., Sholl, L., & Najarian, K. (2019).
   *Classification of prostate cancer whole-slide histopathology images using clustering of persistent homology barcodes*.
   Computerized Medical Imaging and Graphics, 73, 7–17.

9. Li, J., Liu, W., & Xu, D. (2020).
   *Combining radiomics and topological data analysis to classify liver tumors*.
   Artificial Intelligence in Medicine, 107, 101887.

10. Liu, B., Zhang, J., & Wang, Y. (2021).
    *Application of topological data analysis to colorectal cancer histology*.
    Computers in Biology and Medicine, 132, 104323.

11. Oyama, A. et al. (2019).
    *Topological data analysis for the classification of liver tumors by MRI*.
    PLOS ONE, 14(6), e0216726.

12. Qaiser, T. et al. (2019).
    *Fast and accurate tumor segmentation of histology images using persistent homology and deep convolutional features*.
    Medical Image Analysis, 55, 1–14.

13. Somasundaram, E. et al. (2021).
    *Persistent homology-based radiomic feature extraction for prediction of lung cancer survival*.
    Scientific Reports, 11, 13633.

14. Takiyama, R., Oda, M., & Sakamoto, N. (2017).
    *TDA for quantifying the shape of tumor tissue in breast cancer histology*.
    Journal of the Japan Society for Computer Aided Surgery, 19(4), 333–342.

15. Wadhwa, R. R. et al. (2020).
    *Application of persistent homology to enhance mammographic features*.
    Proceedings of the IEEE EMBC, 3789–3792.

[Cripser] A. Bauer. *Cripser: Fast Computation of Cubical Persistence*. Python package, available at https://pypi.org/project/cripser/.

