"""
Project 6: Student Placement Prediction
Model: Logistic Regression
Dataset: Campus Recruitment Dataset
Source: Kaggle - benroshan

Target Variable: status
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set visualization style
sns.set_style("whitegrid")

# Create folder for saving images
IMG_DIR = "Images"
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================
# STREAMLIT TITLE
# ============================================================

st.title("🎓 Student Placement Prediction")
st.subheader("Logistic Regression Model")
st.write("Dataset: Campus Recruitment Dataset")
st.write("Source: Kaggle - benroshan")
st.write("Target Variable: status")


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Make sure Placement_Data_Full_Class.csv is uploaded
# to the same GitHub repository as this file.

df = pd.read_csv("Placement_Data_Full_Class.csv")

st.success("Dataset loaded successfully!")

st.write("Shape of dataset:", df.shape)

st.write("First 5 rows:")
st.dataframe(df.head())

st.write("Column names:")
st.write(df.columns.tolist())

st.write("Data types:")
st.dataframe(df.dtypes.astype(str))


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

st.header("3. Missing Values")

st.write("Missing values:")

st.dataframe(
    df.isnull().sum()
)

# Salary is missing for students who were not placed.
# This is expected in this dataset.

st.write("Salary missing values by placement status:")

st.dataframe(
    df.groupby("status")["salary"]
    .apply(lambda x: x.isnull().sum())
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Drop salary because it is only available after placement.
# Using it would cause data leakage.
#
# Drop sl_no because it is only a serial number and has no
# meaningful predictive value.

df_clean = df.drop(
    columns=["salary", "sl_no"]
)

st.write(
    "Dataset after dropping salary and sl_no:"
)

st.dataframe(
    df_clean.head()
)


# Remove extra spaces from categorical columns

categorical_cols = [
    "gender",
    "workex",
    "hsc_s",
    "degree_t",
    "specialisation",
    "status"
]

for col in categorical_cols:

    df_clean[col] = (
        df_clean[col]
        .astype(str)
        .str.strip()
    )


# Check unique values

st.write(
    "Unique values in categorical columns:"
)

for col in categorical_cols:

    st.write(f"**{col}:**")

    st.write(
        df_clean[col].unique()
    )


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

st.header(
    "5. Exploratory Data Analysis"
)


# ------------------------------------------------------------
# 5.1 Placement by Specialisation and Work Experience
# ------------------------------------------------------------

st.subheader(
    "5.1 Placement by Specialisation and Work Experience"
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


sns.countplot(
    data=df_clean,
    x="specialisation",
    hue="status",
    ax=ax[0]
)

ax[0].set_title(
    "Placement by Specialisation"
)

ax[0].set_xlabel(
    "Specialisation"
)

ax[0].set_ylabel(
    "Number of Students"
)


sns.countplot(
    data=df_clean,
    x="workex",
    hue="status",
    ax=ax[1]
)

ax[1].set_title(
    "Placement by Work Experience"
)

ax[1].set_xlabel(
    "Work Experience"
)

ax[1].set_ylabel(
    "Number of Students"
)


plt.tight_layout()


plt.savefig(
    f"{IMG_DIR}/01_placement_by_spec_workex.png",
    dpi=110,
    bbox_inches="tight"
)


st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# 5.2 Academic Score Distributions
# ------------------------------------------------------------

st.subheader(
    "5.2 Academic Score Distributions"
)


fig, ax = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


score_columns = [
    "ssc_p",
    "hsc_p",
    "degree_p"
]


for i, col in enumerate(score_columns):

    sns.kdeplot(
        data=df_clean,
        x=col,
        hue="status",
        fill=True,
        common_norm=False,
        ax=ax[i]
    )

    ax[i].set_title(
        f"{col} Distribution by Placement Status"
    )

    ax[i].set_xlabel(
        col
    )

    ax[i].set_ylabel(
        "Density"
    )


plt.tight_layout()


plt.savefig(
    f"{IMG_DIR}/02_score_distributions.png",
    dpi=110,
    bbox_inches="tight"
)


st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# 5.3 Correlation Heatmap
# ------------------------------------------------------------

st.subheader(
    "5.3 Correlation Heatmap"
)


# Convert placement status into binary values
# Placed = 1
# Not Placed = 0

df_corr = df_clean.copy()


df_corr["status_bin"] = (
    df_corr["status"] == "Placed"
).astype(int)


score_columns = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p"
]


corr = df_corr[
    score_columns + ["status_bin"]
].corr()


fig, ax = plt.subplots(
    figsize=(8, 6)
)


sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)


ax.set_title(
    "Correlation Heatmap: Academic Scores vs Placement"
)


plt.tight_layout()


plt.savefig(
    f"{IMG_DIR}/03_correlation_heatmap.png",
    dpi=110,
    bbox_inches="tight"
)


st.pyplot(fig)

plt.close(fig)


st.write(
    "Correlation of academic scores with placement:"
)


st.dataframe(
    corr["status_bin"]
    .sort_values(
        ascending=False
    )
)


# ------------------------------------------------------------
# 5.4 MBA Percentage vs Placement
# ------------------------------------------------------------

st.subheader(
    "5.4 MBA Percentage vs Placement"
)


fig, ax = plt.subplots(
    figsize=(7, 5)
)


sns.boxplot(
    data=df_clean,
    x="status",
    y="mba_p",
    ax=ax
)


ax.set_title(
    "MBA Percentage vs Placement Status"
)

ax.set_xlabel(
    "Placement Status"
)

ax.set_ylabel(
    "MBA Percentage"
)


plt.tight_layout()


plt.savefig(
    f"{IMG_DIR}/04_mba_p_vs_status.png",
    dpi=110,
    bbox_inches="tight"
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

st.header(
    "6. Feature Engineering"
)


df_fe = df_clean.copy()


# Create average academic score

df_fe["academic_average"] = df_fe[
    ["ssc_p", "hsc_p", "degree_p"]
].mean(axis=1)


# Convert work experience into binary
# Yes = 1
# No = 0

df_fe["workex_flag"] = (
    df_fe["workex"] == "Yes"
).astype(int)


st.write(
    "Feature engineered dataset:"
)


st.dataframe(
    df_fe.head()
)


# ============================================================
# 7. ENCODE CATEGORICAL VARIABLES
# ============================================================

st.header(
    "7. Encode Categorical Variables"
)


# One-hot encode categorical variables
# Drop first category to avoid dummy variable trap

cat_cols = [
    "gender",
    "hsc_s",
    "degree_t",
    "specialisation"
]


df_encoded = pd.get_dummies(
    df_fe,
    columns=cat_cols,
    drop_first=True
)


# Drop unnecessary columns

df_encoded = df_encoded.drop(
    columns=[
        "ssc_b",
        "hsc_b",
        "workex"
    ]
)


# Convert boolean columns to integers

bool_columns = df_encoded.select_dtypes(
    include="bool"
).columns


df_encoded[bool_columns] = (
    df_encoded[bool_columns]
    .astype(int)
)


st.write(
    "Encoded dataset:"
)


st.dataframe(
    df_encoded.head()
)


# ============================================================
# 8. ENCODE TARGET VARIABLE
# ============================================================

st.header(
    "8. Encode Target Variable"
)


le = LabelEncoder()


df_encoded["status_encoded"] = (
    le.fit_transform(
        df_encoded["status"]
    )
)


st.write(
    "Target encoding:"
)


st.write(
    dict(
        zip(
            le.classes_,
            le.transform(
                le.classes_
            )
        )
    )
)


# ============================================================
# 9. DEFINE FEATURES AND TARGET
# ============================================================

st.header(
    "9. Define Features and Target"
)


X = df_encoded.drop(
    columns=[
        "status",
        "status_encoded"
    ]
)


y = df_encoded[
    "status_encoded"
]


st.write(
    "Feature shape:",
    X.shape
)


st.write(
    "Target shape:",
    y.shape
)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

st.header(
    "10. Train-Test Split"
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


st.write(
    "Training data shape:",
    X_train.shape
)


st.write(
    "Testing data shape:",
    X_test.shape
)


# ============================================================
# 11. FEATURE SCALING
# ============================================================

st.header(
    "11. Feature Scaling"
)


# Scaling is useful for Logistic Regression because
# features have different ranges.

scaler = StandardScaler()


X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)


X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ============================================================
# 12. BUILD LOGISTIC REGRESSION MODEL
# ============================================================

st.header(
    "12. Build Logistic Regression Model"
)


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train model

model.fit(
    X_train_scaled,
    y_train
)


# Make predictions

y_pred = model.predict(
    X_test_scaled
)


st.success(
    "Logistic Regression model trained successfully!"
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

st.header(
    "13. Model Performance"
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


st.write(
    "MODEL PERFORMANCE"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision:.3f}"
    )


with col3:

    st.metric(
        "Recall",
        f"{recall:.3f}"
    )


with col4:

    st.metric(
        "F1 Score",
        f"{f1:.3f}"
    )


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

st.header(
    "14. Classification Report"
)


report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


st.dataframe(
    report_df.round(3)
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

st.header(
    "15. Confusion Matrix"
)


cm = confusion_matrix(
    y_test,
    y_pred
)


fig, ax = plt.subplots(
    figsize=(6, 5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax
)


ax.set_xlabel(
    "Predicted"
)


ax.set_ylabel(
    "Actual"
)


ax.set_title(
    "Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    f"{IMG_DIR}/05_confusion_matrix.png",
    dpi=110,
    bbox_inches="tight"
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# 16. FEATURE IMPORTANCE USING LOGISTIC REGRESSION
# ============================================================

st.header(
    "16. Feature Importance Using Logistic Regression"
)


coef_df = pd.DataFrame({

    "feature":
        X.columns,

    "coefficient":
        model.coef_[0]

})


# Sort by coefficient

coef_df = coef_df.sort_values(
    by="coefficient",
    ascending=False
)


st.write(
    "Feature coefficients:"
)


st.dataframe(
    coef_df,
    use_container_width=True
)


# ============================================================
# 17. TOP FACTORS AFFECTING PLACEMENT
# ============================================================

st.header(
    "17. Top Factors Affecting Placement"
)


st.write(
    "Positive coefficients indicate features that "
    "increase the model's predicted probability "
    "of placement."
)


st.subheader(
    "Top positive factors:"
)


st.dataframe(
    coef_df.head(10)
)


st.subheader(
    "Top negative factors:"
)


st.dataframe(
    coef_df.tail(10)
    .sort_values(
        by="coefficient"
    )
)


# ============================================================
# 18. PLACEMENT RATE ANALYSIS
# ============================================================

st.header(
    "18. Placement Rate Analysis"
)


st.subheader(
    "Placement Rate by Work Experience"
)


workex_placement = pd.crosstab(
    df_clean["workex"],
    df_clean["status"],
    normalize="index"
) * 100


st.dataframe(
    workex_placement.round(2)
)


st.subheader(
    "Placement Rate by Specialisation"
)


specialisation_placement = pd.crosstab(
    df_clean["specialisation"],
    df_clean["status"],
    normalize="index"
) * 100


st.dataframe(
    specialisation_placement.round(2)
)


# ============================================================
# 19. RECOMMENDATIONS
# ============================================================

st.header(
    "19. Placement Improvement Recommendations"
)


st.write(
    """
### 1. Gain relevant work experience

Students with prior work experience may have better
placement outcomes in this dataset.

### 2. Maintain strong academic performance

SSC, HSC, degree and MBA percentages can influence
the model's placement predictions.

### 3. Choose specialisation carefully

Placement outcomes may differ between specialisations
in this dataset.

### 4. Improve employability skills

Students should also focus on communication, aptitude,
interview preparation and technical skills.

### 5. Build practical experience

Internships, projects and industry exposure can improve
job readiness.

### NOTE

These recommendations are based on patterns in the dataset
and Logistic Regression coefficients. They do not prove causation.

Real-world placement decisions also depend on interviews,
skills, company requirements and market conditions.
"""
)


# ============================================================
# 20. DISPLAY SAVED IMAGE LOCATION
# ============================================================

st.header(
    "20. Saved Image Location"
)


st.write(
    "All graphs have been saved successfully in:"
)


st.code(
    IMG_DIR
)
