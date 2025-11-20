---
layout: default
title: Feature Selection
---

# Feature Selection ⚡

Feature selection was a key step in the project to identify the most predictive skills for classifying football players into general categories (**Defender, Midfielder, Forwarder**) and into their **specific positions**.  
By reducing the dimensionality of the dataset, the models could focus on the features that matter most, improving both **performance** and **interpretability**.

---

## 🔹 Initial Feature Set
We started with a comprehensive set of engineered features representing player skills, physical attributes, and mentality metrics, including:

- **Attacking**: crossing, finishing, heading accuracy, short passing, volleys  
- **Skill**: dribbling, curve, long passing, ball control  
- **Movement**: acceleration, sprint speed, agility, balance  
- **Power**: shot power, jumping, stamina, strength  
- **Mentality**: vision, interceptions, positioning  
- **Pace & Shooting metrics** (standardized or transformed)  

A total of **40+ features** were included in the cleaned dataset.

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

