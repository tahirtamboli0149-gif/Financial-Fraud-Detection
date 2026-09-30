import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# reading dataset
df = pd.read_csv("C:\\Users\\Tahir\\Financial-Fraud-Detection\\data\\paysim.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# checking for outliers using IQR method
print("\nOUTLIER CHECK:")
numeric_columns = ["amount", "oldbalanceOrg"]

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"{column}:")
    print("  Lower bound:", lower_bound)
    print("  Upper bound:", upper_bound)
    print("  Number of outliers:", len(outliers))

# removing outliers
original_rows = len(df)
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

removed_rows = original_rows - len(df)
print("\nOUTLIER REMOVAL:")
print("Original rows:", original_rows)
print("Rows removed:", removed_rows)
print("Remaining rows:", len(df))

# selecting features AFTER cleaning
X = df[["type", "amount", "oldbalanceOrg"]]
y = df["isFraud"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# splitting data into train and test part
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData split successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ("numbers", StandardScaler(), ["amount", "oldbalanceOrg"])
    ]
)

print("\nPreprocessing pipeline created successfully!")

# logistic regression model
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)
print("Model trained successfully!")

y_pred_log = logistic_model.predict(X_test)

print("\nLOGISTIC MODEL RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_log, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_log, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

# decision tree model
decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ]
)

decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

print("\nDECISION TREE RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_tree, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_tree, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_tree))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# random forest model
random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ]
)

print("\nTraining Random Forest...")
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

print("\nRANDOM FOREST RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_rf, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# save the random forest model
joblib.dump(random_forest, "src/fraud_model.pkl")
print("\nRandom Forest model saved successfully!")
