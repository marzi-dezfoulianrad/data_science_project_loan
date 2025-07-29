# Loan Approval Prediction Project

This project predicts whether a loan application will be approved based on a set of applicant features. The analysis involves a comprehensive Exploratory Data Analysis (EDA) to uncover insights, followed by the training and evaluation of several machine learning models. The final model, an XGBoost Classifier, demonstrates strong predictive performance.

---

## File Descriptions

* **`EDA.ipynb`**: This notebook contains a detailed exploratory data analysis of the loan dataset. It uses various visualizations like scatter plots, heatmaps, and violin plots, along with statistical tests such as Pearson, Spearman, Chi-Squared, and ANOVA, to investigate the relationships between different features and the loan approval status.

* **`without_EDA.ipynb`**: This notebook represents a baseline approach to modeling. It involves basic data cleaning, one-hot encoding for categorical variables, and label encoding for ordinal features. It then trains and evaluates three models: Logistic Regression, K-Nearest Neighbors (KNN), and an Artificial Neural Network (ANN).

* **`with_EDA.ipynb`**: Building upon the insights from the EDA, this notebook implements more advanced feature engineering, including log transformations and creating interaction terms. It trains a wider range of models: Logistic Regression, KNN, ANN, LightGBM, XGBoost, and a Voting Classifier. The performance of these models is compared, and the best-performing model is saved for future use.

---

## Exploratory Data Analysis (EDA) Summary

The EDA revealed several key factors that influence loan approval:

* **Credit History is Paramount**: The most significant predictor of loan approval. Applicants with a positive credit history have a much higher approval rate.
* **Marital Status and Property Area Matter**: Married applicants and those in semi-urban areas tend to have a higher likelihood of loan approval.
* **Income and Education Influence Loan Amount**: While higher income and being a graduate are associated with larger loan amounts, their direct impact on the approval status itself is less pronounced than the factors above.

---

## Modeling

Two main modeling strategies were employed:

1.  **Baseline Models (`without_EDA.ipynb`)**: This approach used the original features with minimal preprocessing.
2.  **Feature-Engineered Models (`with_EDA.ipynb`)**: This approach incorporated insights from the EDA by creating new features, such as logarithmic transformations of income and loan amounts, and interaction terms like combining marital status with credit history. This strategy led to a significant improvement in model performance.

The models were trained using a pipeline that included scaling numerical features, one-hot encoding categorical features, and applying SMOTE to handle class imbalance.

---

## Results

The models were evaluated based on their **F1-score** on the test set. The **XGBoost Classifier** trained on the feature-engineered dataset from `with_EDA.ipynb` achieved the highest F1-score of **0.90**, making it the best-performing model.

| Model | Feature Set | F1-Score (Test) |
| :--- | :--- | :--- |
| **XGBoost** | **EDA** | **0.900** |
| Logistic Regression | EDA | 0.892 |
| XGBoost | Baseline | 0.884 |
| Voting Classifier | EDA | 0.855 |
| LightGBM | EDA | 0.846 |
| ANN | Baseline | 0.835 |
| Logistic Regression| Baseline | 0.832 |
| LightGBM | Baseline | 0.806 |
| ANN | EDA | 0.780 |
| KNN | EDA | 0.774 |
| KNN | Baseline | 0.756 |

---

## Usage

The best model (`XGB_eda`) has been saved as `best_loan_classifier.joblib`. To use it for predicting new loan applications, you can load the model and use the `predict_loan` function defined in the `with_EDA.ipynb` notebook.

Here is an example of how to make a prediction for a new applicant:

```python
import joblib
import pandas as pd
import numpy as np

# Load the saved model
best_model = joblib.load("../models/best_loan_classifier.joblib")

# New applicant data
new_application = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '1',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

# Feature columns used by the model
feature_cols = [
    'ApplicantIncome_Log', 'CoapplicantIncome_Log', 'LoanAmount_Log',
    'Total_Income', 'Loan_to_Income_Ratio', 'Loan_per_Month',
    'Married_Credit', 'Edu_Loan', 'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed', 'Property_Area', 'Credit_History'
]

# Prediction function (assuming it's defined as in the notebook)
def predict_loan(new_application_dict, model_pipeline, feature_cols):
    df_new = pd.DataFrame([new_application_dict])

    # Create engineered features
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        df_new[f'{col}_Log'] = np.log1p(df_new[col])
    df_new['Total_Income'] = df_new['ApplicantIncome'] + df_new['CoapplicantIncome']
    df_new['Loan_to_Income_Ratio'] = df_new['LoanAmount'] / df_new['Total_Income']
    df_new['Loan_per_Month'] = df_new['LoanAmount'] / (df_new['Loan_Amount_Term'] / 12)
    df_new['Married_Credit'] = df_new['Married'].map({'Yes':1, 'No':0}) * df_new['Credit_History']
    df_new['Edu_Loan'] = df_new['Education'].map({'Graduate':1, 'Not Graduate':0}) * df_new['LoanAmount']

    # Ensure all necessary feature columns are present
    df_new = df_new.reindex(columns=feature_cols, fill_value=0)

    pred = model_pipeline.predict(df_new)[0]
    prob = model_pipeline.predict_proba(df_new)[0, 1]

    print(f"Prediction: {'Approved' if pred else 'Not Approved'} (Confidence: {prob:.2%})")

# Make the prediction
predict_loan(new_application, best_model, feature_cols)
# Expected Output: Prediction: Approved (Confidence: 60.23%)
