# 💼 Employee Salary Prediction

> Predicting whether an employee’s salary is above or below a certain threshold using machine learning algorithms.

---

## 📌 **Table of Contents**
- [🎯 Objective](#-objective)
- [🔍 Methodology](#-methodology)
- [🗂️ Dataset](#️-dataset)
- [⚙️ Tools & Libraries](#️-tools--libraries)
- [📈 Results](#-results)
- [✅ Conclusion](#-conclusion)

---

## 🎯 **Objective**

The main goal of this project is to build a robust machine learning model that can predict whether an employee earns more than a defined salary threshold based on demographic and professional attributes.

---

## 🔍 **Methodology**

1. **Data Loading**  
   - Loaded the dataset `adult 3.csv` containing employee details.

2. **Data Cleaning & Preprocessing**  
   - Checked for missing values and handled them using mean, median, mode, or arbitrary values.
   - Encoded categorical variables using `LabelEncoder`.
   - Standardized numerical features with `StandardScaler`.

3. **Model Building**  
   - Multiple classification models were implemented:
     - Logistic Regression
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - AdaBoost Classifier
     - Naive Bayes
     - Decision Tree
     - K-Nearest Neighbors
     - Support Vector Machine (SVM)

4. **Model Evaluation**  
   - Evaluated models using metrics like:
     - Accuracy Score
     - Confusion Matrix
     - ROC AUC Score
     - Classification Report

---

## 🗂️ **Dataset**

The dataset used is the **Adult Census Income Dataset**, which includes attributes such as:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Hours per week
- Native country

---

## ⚙️ **Tools & Libraries**

- Python 
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📈 **Results**

- Multiple models were compared to identify the best performing classifier.
- Ensemble models like Random Forest & Gradient Boosting performed best.
- Evaluation metrics like accuracy and ROC AUC Score were used to select the best model for predicting employee salaries.

---

## ✅ **Conclusion**

This project demonstrates the end-to-end process of:
- Data preprocessing  
- Building and training multiple models  
- Evaluating their performance  
- Selecting the best performing model for salary prediction.

This solution can be extended or fine-tuned further with hyperparameter tuning or using more advanced models.

---
## 📎 **Author**
**Aneena Jose Thaliath**    
📧 aneenajosethaliath@gmail.com
