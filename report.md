# 📝 Traffic Accident Severity Prediction - Final Report

## ✅ Project Overview

This project explores the development of a machine learning model to **predict the severity of road accidents** using data collected on road conditions, environmental factors, driver demographics, and vehicle movement. Accurate prediction of accident severity supports proactive safety interventions and better emergency planning.

---

## 🎯 Problem Statement

Given historical accident data, our goal is to classify each incident into one of three severity levels:

- **0** = Slight Injury
- **1** = Serious Injury
- **2** = Fatal Injury

By leveraging driver attributes, road conditions, light/weather states, and accident causes, the system aims to assist in anticipating accident outcomes and preventing fatalities.

---

## 📂 Dataset

- **Source**: RTA Accident Dataset (via Kaggle)
- **Preprocessed File**: `cleaned.csv`
- **Total Samples**: 12,316
- **Features Used**: 97 (after one-hot encoding of 14 categorical variables)
- **Target Variable**: `Accident_severity` (multiclass: 0, 1, 2)

---

## 🔧 Feature Engineering

- One-hot encoding applied to:
  - `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Driving_experience`,
  - `Lanes_or_Medians`, `Types_of_Junction`, `Road_surface_type`, `Light_conditions`,
  - `Weather_conditions`, `Type_of_collision`, `Vehicle_movement`, `Pedestrian_movement`,
  - `Cause_of_accident`, `Vehicle_driver_relation`
- Training/testing split: **80/20**
- Stratification ensured balanced class distribution

---

## 🧠 Models Trained & Evaluated

| Model               | Accuracy   | Precision | Recall     | F1 Score   |
| ------------------- | ---------- | --------- | ---------- | ---------- |
| **Random Forest**   | 0.8405     | 0.7545    | 0.8405     | **0.7775** |
| Logistic Regression | **0.8458** | 0.7153    | **0.8458** | 0.7751     |
| Decision Tree       | 0.7216     | 0.7462    | 0.7216     | 0.7333     |

---

## 🏆 Best Model: **Random Forest**

Although Logistic Regression had slightly higher accuracy, **Random Forest outperformed** on overall **F1 Score** and **precision**, making it the most balanced and reliable choice for predicting accident severity across all classes.

### ✅ Why Random Forest?

- Handles feature interactions and non-linear patterns well
- Robust to noise and outliers
- High recall ensures fewer severe accidents are missed

---

## 📊 Visual Comparison

> See `week5_evaluation_and_reporting.ipynb` for detailed plots of accuracy, precision, recall, and F1-score across all models.

---

## 👨‍💻 Technologies Used

- Python 3.11.9
- pandas, scikit-learn, seaborn, matplotlib
- Jupyter Notebook
- GitHub for version control

---

## 👥 Team Members

- **Mark Brian** – Team Lead, Integration & Reporting
- Member A – Data Preprocessing & Cleaning
- Member B – Exploratory Data Analysis
- Member C – Feature Engineering
- Member D – Model Development
- Member E – Evaluation & Visualization

---

## 📌 Key Insights

- Slight injuries were the most frequent, but fatal accidents still require attention.
- Poor lighting, road surface type, and collision angle significantly impact severity.
- Tree-based models performed better due to the complexity of interactions.

---

## 📦 Repository & Submission

✅ [GitHub Repository Link](https://github.com/Markbkiunga/traffic-accident-severity-prediction)
📎 Final Deliverables:

- `cleaned.csv`
- Notebooks: `week1_intro.ipynb` → `week5_evaluation_and_reporting.ipynb`
- `report.md` (this file)
- `README.md`

---
