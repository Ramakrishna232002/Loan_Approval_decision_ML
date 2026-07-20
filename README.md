# Loan Approval Prediction - End-to-End Machine Learning Project

## Project Overview

This project implements a complete Machine Learning lifecycle for predicting loan approval status.

The main objective was to understand and implement an industry-style ML workflow:

- Data Understanding
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Missing Value Handling
- Duplicate Checking
- Outlier Analysis
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Comparison
- Hyperparameter Tuning
- Cross Validation
- Pipeline Creation
- Model Saving

---

# Problem Statement

Banks receive many loan applications and need an automated system to predict whether a loan should be approved or rejected.

This project builds a binary classification model that predicts:


---

# Dataset Information

Dataset contains 381 records and 13 columns.

## Features

| Feature | Description |
|---|---|
| Loan_ID | Unique loan identifier |
| Gender | Applicant gender |
| Married | Marriage status |
| Dependents | Number of dependents |
| Education | Education qualification |
| Self_Employed | Employment status |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Co-applicant income |
| LoanAmount | Loan amount requested |
| Loan_Amount_Term | Loan repayment period |
| Credit_History | Credit history |
| Property_Area | Property location |
| Loan_Status | Target variable |

---

# 1. Data Loading and Understanding

Dataset was loaded using Pandas.

```python
df = pd.read_csv("loan.csv")

                 END-TO-END MACHINE LEARNING WORKFLOW
                 =====================================


1. DATA LOADING
================

Load dataset using Pandas

        |
        ↓

Read CSV file

        |
        ↓

Check initial data


Operations:

- df.head()
- df.shape()
- df.info()
- df.describe()
- df.columns



                 |
                 ↓


2. EXPLORATORY DATA ANALYSIS (EDA)
===================================

Understand the dataset


Analyze:

- Target distribution
- Feature distribution
- Categorical features
- Numerical features
- Relationship between features and target


Examples:

Loan_Status distribution

Gender distribution

Credit_History distribution

Income distribution

Loan Amount distribution


Visualization:

- Count plots
- Histograms
- Box plots
- Correlation heatmap



                 |
                 ↓


3. DATA CLEANING
=================


3.1 Missing Value Detection

Check:

df.isnull().sum()


Found missing values:

Categorical:

- Gender
- Dependents
- Self_Employed


Numerical:

- Loan_Amount_Term
- Credit_History



                 |
                 ↓


3.2 Duplicate Checking


Check:

df.duplicated().sum()


Remove duplicates if present.



                 |
                 ↓


3.3 Data Type Checking


Verify:

- Object columns
- Integer columns
- Float columns


Convert data types if required.



                 |
                 ↓


4. OUTLIER ANALYSIS
====================


Check numerical features:

- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term


Method:

IQR Method


Calculate:

Q1

Q3

IQR = Q3 - Q1


Lower Bound:

Q1 - 1.5 * IQR


Upper Bound:

Q3 + 1.5 * IQR



Decision:

Do not remove outliers because:

- Financial extremes may contain information
- Tree models handle outliers well



                 |
                 ↓


5. FEATURE ENGINEERING
=======================


5.1 Separate Features and Target


X:

All independent features


y:

Loan_Status



Example:


X = df.drop("Loan_Status")

y = df["Loan_Status"]



                 |
                 ↓


5.2 Target Encoding


Convert:


Y  → 1

N  → 0



                 |
                 ↓


5.3 Identify Feature Types


Numerical Features:


- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term



Categorical Features:


- Gender
- Married
- Dependents
- Education
- Self_Employed
- Property_Area
- Credit_History



                 |
                 ↓


6. TRAIN TEST SPLIT
====================


Split data:


80% Training Data

20% Testing Data


Using:


train_test_split()


Parameters:


random_state=42

stratify=y



                 |
                 ↓


7. BUILD PREPROCESSING PIPELINE
================================


Create separate pipelines.


------------------------------------------------

Numerical Pipeline


Input:

Numerical Features


Steps:


Missing Values

        |
        ↓

Median Imputation


------------------------------------------------


Categorical Pipeline


Input:

Categorical Features


Steps:


Missing Values

        |
        ↓

Most Frequent Imputation


        |
        ↓

One Hot Encoding



------------------------------------------------



                 |
                 ↓


8. COLUMN TRANSFORMER
======================


Combine both pipelines:


ColumnTransformer


Flow:


Numerical Columns

        |
        ↓

Numerical Pipeline


+
 

Categorical Columns

        |
        ↓

Categorical Pipeline



                 |
                 ↓


9. MODEL SELECTION
===================


Train multiple models:


Baseline:

Logistic Regression


Distance Based:

KNN

SVM


Tree Based:

Random Forest


Boosting:

XGBoost



Compare:

- Accuracy
- Precision
- Recall
- F1 Score



                 |
                 ↓


10. FINAL MODEL SELECTION
==========================


Selected:


XGBoost Classifier



Parameters tuned:


n_estimators

learning_rate

max_depth


Example:


XGBClassifier(

n_estimators=200,

learning_rate=0.2,

max_depth=5,

random_state=42

)



                 |
                 ↓


11. CREATE COMPLETE ML PIPELINE
================================


Combine:


Preprocessor

+

Model



Final pipeline:


Raw Data

    |
    ↓

Column Transformer

    |
    ↓

Imputation

    |
    ↓

Encoding

    |
    ↓

XGBoost Model

    |
    ↓

Prediction



                 |
                 ↓


12. MODEL TRAINING
===================


Train pipeline:


model_pipeline.fit(

X_train,

y_train

)



Pipeline automatically:


Learns median values

Learns most frequent categories

Learns encoding mapping

Trains model



                 |
                 ↓


13. MODEL EVALUATION
=====================


Predict:


y_pred = model_pipeline.predict(X_test)



Evaluate:


Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Classification Report



                 |
                 ↓


14. CROSS VALIDATION
=====================


Check model stability:


cross_val_score()


Example:


cv=5


Evaluate:


Average Accuracy

Average F1 Score



                 |
                 ↓


15. FINAL MODEL TRAINING
=========================


Use final selected pipeline:


model_pipeline


Ready for production.



                 |
                 ↓


16. MODEL SERIALIZATION
========================


Save complete pipeline:


Using Joblib:



joblib.dump(

model_pipeline,

"loan_approval_model.pkl"

)



Saved file contains:


----------------------

Preprocessing

+

Encoder

+

XGBoost Model

----------------------



                 |
                 ↓


17. MODEL LOADING
==================


Later:


model = joblib.load(

"loan_approval_model.pkl"

)



Now model can directly accept:

Raw customer data


and return:


Loan Approved

or

Loan Rejected



                 |
                 ↓


                 END
,,,