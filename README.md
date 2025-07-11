# Predicting Financial Health and Risk Profiles

This project aims to classify individuals into financial health categories—**Healthy**, **Moderate Risk**, or **High Risk**—based on synthetic data encompassing demographic, financial, life event, and economic variables. This classification can be useful for applications in fintech, credit scoring, and personalized financial services.

---

## 📌 Objective

Build a machine learning model to predict the financial risk profile of individuals using supervised learning methods, data preprocessing techniques, and performance evaluation metrics.

---

## 🗂️ Dataset Overview

- **Synthetic dataset** with the following feature categories:
  - Demographic: Age, Gender, Marital Status, etc.
  - Financial: Income, Debt, Expenses, Savings, Credit Score
  - Life Events: Employment Change, Medical Emergency, etc.
  - Macroeconomic: Inflation Rate, Economic Indicator Scores

- **Target variable**:
  - `Financial Health`: One of `Healthy`, `Moderate Risk`, or `High Risk`

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- LightGBM

---

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (Per class and Macro Average)
- Confusion Matrix
- ROC-AUC (if applicable)

---

## 🛠️ Features and Workflow

1. **Data Cleaning**:
   - Handling missing values
   - Removing duplicates
   - Type conversions

2. **Exploratory Data Analysis (EDA)**:
   - Distribution plots
   - Correlation heatmaps

3. **Feature Engineering**:
   - One-hot encoding for categorical variables
   - Scaling numeric features

4. **Model Training and Evaluation**:
   - Train-test split
   - Model comparison on evaluation metrics

5. **Model Performance Analysis**:
   - Discussion on class imbalance and possible strategies (e.g., SMOTE, class weighting)

---

## 🧪 Key Insights

- The model faced challenges with **class imbalance**, affecting precision and recall for the `Healthy` and `High Risk` classes.
- Models like Random Forest and XGBoost performed better in generalization compared to logistic regression or SVM.

---

## 🔧 Future Improvements

- Use ensemble stacking to boost performance
- Incorporate SMOTE or ADASYN to handle class imbalance
- Deploy model using Flask or FastAPI
- Monitor real-time predictions and drift

