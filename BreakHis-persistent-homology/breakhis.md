---
layout: default
title: BreakHis data
---
# **Data Preprocessing**

## About the dataset

The **BreakHis** (Breast Cancer Histopathological Image) dataset is a publicly available collection of microscopic breast tissue images used for developing machine-learning models for tumour classification. It contains **7,909 RGB histology images** from **82 patients**, labelled as **benign** or **malignant**, with each class further divided into several subtypes.

Images were captured using optical microscopy at **40×, 100×, 200×, and 400× magnifications**, offering multiscale structural information. Each image is **700×460 pixels (JPEG)** and reflects natural variability in staining, colour, and texture—making BreakHis a realistic and challenging benchmark for medical-image analysis and topological data analysis (TDA).


## **Stain Normalization (StainTools)**
Histopathology images often differ in stain appearance due to lab conditions and scanner variability.  
To reduce this domain shift, **StainTools** was used to apply stain normalization.

Steps:
1. Chose a reference image with a desirable stain profile.
2. Applied **Macenko**  normalization to all images.
3. Produced a dataset with consistent stain intensity and structure.

![Example of stain normalisation at 40× magnification. Left: original image; right: normalised image produced with StainTools (Macenko method)](portfolio/BreakHis-persistent-homology/assets/images/Screenshot%202025-11-22%20104316.png)


This step ensures that topological features are not impacted by stain inconsistencies.

---

## Class Imbalance & Data Augmentation

After stain normalisation, the BreakHis dataset showed a notable class imbalance, with **benign (1,370)** images outnumbering **malignant (625)**. To reduce bias and improve model generalisation, we applied **data augmentation** to the minority class during **training only**.

Augmentations included:
- small-angle rotations  
- horizontal/vertical flips  
- isotropic zooming  
- brightness jitter  

All transformations were kept mild to preserve tissue morphology, and no augmentation was applied to the test set to prevent data leakage. This resulted in a more balanced training distribution while retaining the original class proportions for evaluation.


## **Conversion to Grayscale (Red Channel Extraction)**
Persistent homology (PH) requires scalar-valued inputs. Because H&E-stained tissue images have the strongest structural contrast in the **red channel**, the red channel was extracted from each RGB image.

- Extracted the red channel from every image.
- Used it as the grayscale representation for PH computations.
---

## **Final Output**
After preprocessing:
- All images were stain-normalized.
- Red-channel grayscale images were generated.
- These standardized images were used as inputs for persistent homology extraction.

