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

import cripser

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

![Example image](assets/images/Screenshot%202025-11-22%20160454.png)

---

# Vectorisation of Persistence Diaagrams

The issue is that persistence barcodes are not in an immediately algorithm-friendly form, despite capturing rich topological information. Many machine learning algorithms are relying on vectorized database to handle large dataset and allow fast computation. Vectorized dataset allow us to apply the law of large numbers or central-limit theorem, hypothesis testing, direct comparison using linear algebra and many more. 

The Persistence Image (PI) approach was introduced by Henry Adams et al [1]. In this method, the authors convert a persistence diagram (PD) into a finite-dimensional vector representation, which they termed a persistence image.

The code block below demonstrates vectorisation of the persistence diagrams into persistence images using **persim** library. 


```python

# Vectorising PDs into Persistence Images using PersistenceImager function from persim library

import persim
from persim import plot_diagrams, PersistenceImager
from tqdm import tqdm

persistent_images_h0 = []  # dimension 0, i.e. connected components
persistent_images_h1 = []  # dimension 1, i.e. loops and holes

for i, pd_diagram in enumerate(tqdm(pds, desc="Generating persistence images")):
    try:
        pimgr = PersistenceImager(pixel_size=0.01)
        pimgr.kernel_params = {'sigma': 0.001}

        # --- H0 ---
        pimgr.fit(pd_diagram[0])
        img0 = pimgr.transform(pd_diagram[0])

        # --- H1 ---
        pimgr.fit(pd_diagram[1])
        img1 = pimgr.transform(pd_diagram[1])

        # Append to resective arrays
        persistent_images_h0.append(img0)
        persistent_images_h1.append(img1)

```

![Example image](assets/images/Screenshot%202025-11-22%20161113.png)


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

