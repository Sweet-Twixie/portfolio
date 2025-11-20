---
layout: default
title: Feature Engineering
---

# Feature Engineering ⚙️

Feature engineering was a critical step in transforming raw football data into meaningful inputs for machine learning models. The dataset contained **20,000+ players** with categorical, numerical, and text-based attributes. My goal was to reshape this data so the models could better capture player roles and performance.

---

## 🔹 Encoding Player Positions
Each player could play in multiple positions, stored as text like `"CF, ST, LM"`. To make this usable, I converted the `"player_positions"` field into structured features:

- **General categories**:  
  - **Defender** → CB, RB, LB, CDM  
  - **Midfielder** → CM, LM, RM, CAM  
  - **Forwarder** → ST, LW, RW, CF, LWB, RWB  
  - **Goalkeeper** → GK  

- **Specific positions**: A binary column for each role (e.g., CB=1, ST=0).  

👉 Goalkeepers were excluded from further modeling since they can be trivially identified from their unique goalkeeping stats.

This gave every player a clear **multi-label representation**:  
- General role (Defender/Midfielder/Forwarder).  
- All specific positions they can play.  

---

## 🔹 Handling Left/Right Roles
Some positions appear in left/right pairs (e.g., **LB/RB**, **LW/RW**). To reduce redundancy, I tested combining them into aggregated features:

- `LW/RW`  
- `LB/RB`  
- `LM/RM`  
- `LWB/RWB`  

Values could be:  
- `0` → player doesn’t play either role.  
- `1` → player plays one of them.  
- `2` → player can play both.  

However, since very few players had value `2`, I ultimately kept **left/right roles separate** to preserve detail.

---

## 🔹 Numerical Feature Engineering
Beyond positions, the dataset contained **40+ numerical attributes** describing player skills (e.g., dribbling, crossing, stamina, vision). These raw values required careful preprocessing to ensure they were both interpretable and suitable for machine learning models.

### Step 1: Distribution Analysis
- Plotted **distributions for each feature** to understand shape, skewness, and outliers.  
- Identified features with strong skew (e.g., power attributes), and others with **multi-modal distributions** (e.g., defending skills had two peaks → possibly reflecting defenders vs. non-defenders).  

### Step 2: Transformations
- For skewed distributions, applied different transformations:
  - **Logarithmic**
  - **Exponential**
  - **Power**
  - **Box-Cox**  
- Chose the most appropriate transformation **visually**, by comparing before/after plots.

### Step 3: Normalization & Standardization
- Applied **standardization** (mean=0, variance=1) to features for comparability.  
- Applied **normalization** where necessary to scale attributes into a consistent range.  

### Step 4: Feature Categorization
- Some distributions suggested **natural groups** (e.g., defending had “two bumps”).  
- Created **binned categories** for features like `attacking_crossing` to simplify interpretation.  

---

## 🔹 Numerical Feature Engineering Examples

Below are examples of feature transformations. Each pair shows the **raw distribution (before)** and the **transformed distribution (after)**.

---

### ⚽ Attacking
<div style="display: flex; justify-content: space-between;">
  <img src="assets/images/Before Attacking.png" alt="Before Attacking" width="48%">
  <img src="assets/images/After Attacking.png" alt="After Attacking" width="48%">
</div>

---

### ⚽ Dribbling
<div style="display: flex; justify-content: space-between;">
  <img src="assets/images/Before Dribbling.png" alt="Before Dribbling" width="48%">
  <img src="assets/images/After Dribbling.png" alt="After Dribbling" width="48%">
</div>

---

### ⚽ Wage
<div style="display: flex; justify-content: space-between;">
  <img src="assets/images/Before wage.png" alt="Before Wage" width="48%">
  <img src="assets/images/After wage.png" alt="After Wage" width="48%">
</div>

---

### ⚽ Defending
<div style="display: flex; justify-content: space-between;">
  <img src="assets/images/before defender.png" alt="Before Defending" width="48%">
  <img src="assets/images/after defender.png" alt="After Defending" width="48%">
</div>

---

## ✅ Outcome
After feature engineering, the dataset was clean, structured, and model-ready:  
- **Categorical text fields** (positions) were transformed into interpretable binary/multi-label features.  
- **Numerical attributes** were standardized, normalized, or transformed for stability.  
- **Goalkeepers** were excluded from outfield classification, ensuring focus on the three main roles: **Defender, Midfielder, Forwarder**.  

This solid foundation allowed for meaningful **feature selection** and accurate **multi-label classification** in the later modeling stages.


