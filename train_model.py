import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# SMART LOAN AI 2.0
# MODEL TRAINING SYSTEM
# ============================================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("        SMART LOAN AI 2.0 MODEL TRAINING")
print("=" * 60)
print()

np.random.seed(42)

# Number of synthetic records
n_records = 12000

print(f"Generating {n_records:,} realistic banking records...")
print()


# ============================================================
# 1. GENERATE BASIC APPLICANT DATA
# ============================================================

genders = np.random.choice(
    ["Male", "Female"],
    size=n_records,
    p=[0.60, 0.40]
)

married = np.random.choice(
    ["Yes", "No"],
    size=n_records,
    p=[0.65, 0.35]
)

dependents = np.random.choice(
    ["0", "1", "2", "3+"],
    size=n_records,
    p=[0.50, 0.20, 0.15, 0.15]
)

education = np.random.choice(
    ["Graduate", "Not Graduate"],
    size=n_records,
    p=[0.80, 0.20]
)

self_employed = np.random.choice(
    ["Yes", "No"],
    size=n_records,
    p=[0.15, 0.85]
)

property_area = np.random.choice(
    ["Urban", "Semiurban", "Rural"],
    size=n_records,
    p=[0.30, 0.40, 0.30]
)

applicant_age = np.random.randint(
    21,
    66,
    size=n_records
)


# ============================================================
# 2. GENERATE REALISTIC MONTHLY INCOME
# ============================================================

applicant_income = np.random.lognormal(
    mean=np.log(35000),
    sigma=0.55,
    size=n_records
)

applicant_income = np.clip(
    applicant_income,
    10000,
    250000
).astype(int)


coapplicant_income = np.where(
    np.random.rand(n_records) > 0.40,
    np.random.lognormal(
        mean=np.log(20000),
        sigma=0.60,
        size=n_records
    ),
    0
)

coapplicant_income = np.clip(
    coapplicant_income,
    0,
    150000
).astype(int)


# ============================================================
# 3. TOTAL MONTHLY INCOME
# ============================================================

total_income = applicant_income + coapplicant_income


# ============================================================
# 4. GENERATE LOAN AMOUNT
# ============================================================

# Loan amount stored in THOUSANDS because the original
# ML model uses this format.

loan_amount = (
    total_income *
    np.random.uniform(3.0, 7.0, size=n_records)
    / 1000
)

loan_amount = np.clip(
    loan_amount,
    50,
    1500
).astype(int)


# ============================================================
# 5. LOAN TERM
# ============================================================

loan_term = np.random.choice(
    [12, 24, 36, 48, 60, 84, 120, 180, 240, 360],
    size=n_records,
    p=[
        0.02,
        0.04,
        0.06,
        0.07,
        0.12,
        0.10,
        0.15,
        0.14,
        0.14,
        0.16
    ]
)


# ============================================================
# 6. CREDIT HISTORY
# ============================================================

credit_history = np.random.choice(
    [1.0, 0.0],
    size=n_records,
    p=[0.82, 0.18]
)


# ============================================================
# 7. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Applicant_Age": applicant_age
})


# ============================================================
# 8. REALISTIC LOAN ELIGIBILITY LOGIC
# ============================================================

# IMPORTANT:
# ApplicantIncome and CoapplicantIncome are MONTHLY income.
# Therefore we DO NOT divide total income by 12.

monthly_income = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)


# Approximate monthly principal repayment.
# LoanAmount is in thousands.

monthly_principal = (
    df["LoanAmount"] * 1000
) / df["Loan_Amount_Term"]


# Debt-to-income ratio

dti_ratio = (
    monthly_principal /
    monthly_income
)


# ============================================================
# 9. ELIGIBILITY SCORE
# ============================================================

score = np.zeros(n_records)


# Credit history
score += np.where(
    df["Credit_History"] == 1.0,
    45,
    -30
)


# Education
score += np.where(
    df["Education"] == "Graduate",
    10,
    0
)


# Age
score += np.where(
    df["Applicant_Age"].between(25, 55),
    8,
    0
)


# Marriage
score += np.where(
    df["Married"] == "Yes",
    4,
    0
)


# Employment
score += np.where(
    df["Self_Employed"] == "No",
    4,
    0
)


# Property area
score += np.where(
    df["Property_Area"] == "Urban",
    5,
    np.where(
        df["Property_Area"] == "Semiurban",
        3,
        0
    )
)


# Income strength

income_strength = np.clip(
    monthly_income / 50000,
    0,
    2
)

score += income_strength * 10


# DTI penalty

score -= np.clip(
    dti_ratio * 30,
    0,
    30
)


# Small natural variation
score += np.random.normal(
    0,
    5,
    n_records
)


# ============================================================
# 10. LOAN STATUS
# ============================================================

df["Loan_Status"] = np.where(
    score >= 30,
    "Y",
    "N"
)


# ============================================================
# 11. BALANCE DATASET
# ============================================================

# Print distribution before training

print("Loan Status Distribution:")
print(
    df["Loan_Status"].value_counts()
)
print()


# ============================================================
# 12. SAVE DATASET
# ============================================================

df.to_csv(
    "dataset/loan_data.csv",
    index=False
)

print("Dataset saved successfully:")
print("dataset/loan_data.csv")
print()


# ============================================================
# 13. MACHINE LEARNING PREPROCESSING
# ============================================================

df_ml = df.copy()

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

encoders = {}


for col in categorical_cols:

    encoder = LabelEncoder()

    df_ml[col] = encoder.fit_transform(
        df_ml[col].astype(str)
    )

    encoders[col] = encoder


# Target encoding

df_ml["Loan_Status"] = (
    df_ml["Loan_Status"]
    .map({
        "Y": 1,
        "N": 0
    })
)


# ============================================================
# 14. FEATURES AND TARGET
# ============================================================

X = df_ml.drop(
    "Loan_Status",
    axis=1
)

y = df_ml["Loan_Status"]


print("Features used by model:")
print(list(X.columns))
print()


# ============================================================
# 15. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    f"Training records : {len(X_train):,}"
)

print(
    f"Testing records  : {len(X_test):,}"
)

print()


# ============================================================
# 16. RANDOM FOREST MODEL
# ============================================================

print("Training Random Forest model...")
print()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 17. MODEL EVALUATION
# ============================================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print()
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Rejected",
            "Approved"
        ]
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print()


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_data = {
    "model": model,
    "encoders": encoders,
    "features": list(X.columns)
}


with open(
    "models/loan_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model_data,
        file
    )


print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print()
print(
    "File: models/loan_model.pkl"
)

print()
print("SMART LOAN AI 2.0 is ready!")