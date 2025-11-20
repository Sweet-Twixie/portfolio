---
layout: default
title: Football Player Position Prediction
---

# Football Player Position Prediction ⚽

A machine learning project predicting **football player positions** based on their skills, physical attributes, and mentality metrics — using data from nearly **20,000 professional players**.  

The goal: help **clubs, scouts, and analysts** identify versatile players, optimize squad building, and support smarter recruitment and development strategies.

---

## Project Highlights
- **End-to-end ML pipeline**: data cleaning, feature engineering, feature selection, and classification  
- **Hierarchical multi-label classification**:  
  - Broad categories: **Defender, Midfielder, Forward** (Goalkeepers excluded early)  
  - Specific on-field positions within each category  
- **Feature engineering**: transformations, binning, and standardization to extract the most predictive signals  
- **Feature selection**: Random Forest, RFE, and Gradient Boosting used to identify the top features  
- **Models tested**: Logistic Regression, Random Forest, Classifier Chains and Neural Networks  
- **Best performance**: Neural Networks (for general categories & most positions) + Classifier Chains (for Midfielders)  
- **Final ensemble**: combined models to achieve **~75% relaxed accuracy** across 20+ multi-label positions  

---

## Key Takeaway
This project demonstrates how machine learning can effectively classify football players into **both general roles and specific field positions**, opening opportunities for real-world applications in **player scouting, transfer market analysis, and tactical planning**.

## Explore Sections
- [EDA and Feature Engineering](feature-engineering.html) – skill transformations and encoding
- [Feature Selection](feature-selection.html) – methods and chosen features
- [ML Classification & Ensemble](ml-classification.html) – models, hierarchical classifier, and results

## Repository
The full code and dataset processing are available here:
[GitHub Repository](https://github.com/Sweet-Twixie/football.git)


