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
- **Original Samples**: 12,316
- **Post-SMOTE Samples**: 31,245 (Balanced)
- **Features Used**: 97 (after one-hot encoding of 14 categorical variables)
- **Target Variable**: `Accident_severity` (multiclass: 0, 1, 2)

---

## ⚠️ Class Imbalance & Oversampling

### 🎯 Original Class Distribution:

- **Fatal Injury (2)**: 10,415
- **Serious Injury (1)**: 1,743
- **Slight Injury (0)**: 158

This imbalance led to biased predictions toward fatal injuries.

### ✅ SMOTE Applied:

We applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the classes:

- All classes resampled to **10,415** observations
- Total dataset after oversampling: **31,245 records**
- Train-test split: **80/20 stratified**

---

## 🔧 Feature Engineering

- One-hot encoding applied to:

  ```
  Age_band_of_driver, Sex_of_driver, Educational_level, Driving_experience,
  Lanes_or_Medians, Types_of_Junction, Road_surface_type, Light_conditions,
  Weather_conditions, Type_of_collision, Vehicle_movement, Pedestrian_movement,
  Cause_of_accident, Vehicle_driver_relation
  ```

- All features converted to numeric
- Split into training (24,996 samples) and testing sets (6,249 samples)

---

## 🧠 Models Trained & Evaluated

| Model               | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest**   | **0.8896** | **0.8897** | **0.8896** | **0.8890** |
| Decision Tree       | 0.8317     | 0.8335     | 0.8317     | 0.8284     |
| Logistic Regression | 0.7323     | 0.7256     | 0.7323     | 0.7268     |

---

## 🏆 Best Model: **Random Forest (After SMOTE)**

The **Random Forest** model, trained on the balanced dataset, achieved the best overall performance across **accuracy, precision, recall, and F1 score**.

### ✅ Why Random Forest?

- Effectively handles categorical and numerical features
- Captures complex patterns and interactions
- Resistant to overfitting, especially with many trees
- Balanced recall across all severity classes

---

## 📊 Visual Evaluation

Bar plots comparing **accuracy, precision, recall, and F1 score** across all models are available in:

📓 `week5_evaluation_and_reporting.ipynb`

---

## 👨‍💻 Technologies Used

- Python 3.11
- Libraries: pandas, scikit-learn, imbalanced-learn, seaborn, matplotlib, joblib
- Jupyter Notebook
- Streamlit (for deployment)
- GitHub (version control)

---

## 🧪 Deployment & Testing

The final Random Forest model and feature list were saved using `joblib` and deployed via a **Streamlit web application**. Users can input accident conditions and receive real-time predictions with class probabilities.

---

## 👥 Team Members

- **Mark Brian** – Team Lead, Integration & Reporting
- Michelle Kere – Data Preprocessing & Cleaning
- Ridwan Bare – Exploratory Data Analysis
- Mitari – Feature Engineering
- Precious Imali and Joy Splendor – Model Development, Evaluation & Visualization

---

## 📌 Key Insights

- Class imbalance significantly hindered original model performance
- SMOTE dramatically improved the model’s ability to predict **serious and slight injuries**
- Fatal injuries are still dominant in the original data, emphasizing a need for better road policies
- Poor visibility, road type, and driver behavior strongly influence accident severity

---

## 📦 Repository & Submission

✅ [GitHub Repository Link](https://github.com/Markbkiunga/traffic-accident-severity-prediction)  
📎 Final Deliverables:

- `cleaned.csv`
- Notebooks: `week1_intro.ipynb` → `week5_evaluation_and_reporting.ipynb`
- `report.md` (this file)
- `README.md`
- `streamlit_app.py`
- `models/random_forest_model.pkl`
- `models/feature_columns.pkl`
