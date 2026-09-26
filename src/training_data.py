import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import   confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



#reading dataset
df = pd.read_csv("C:\\Users\\Tahir\\Financial-Fraud-Detection\\data\\paysim.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())


#selecting features
X = df[["type", "amount", "oldbalanceOrg"]]
y = df["isFraud"]

# Targeting variable
y = df["isFraud"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

#splitting data into train and test part
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData split successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ("numbers", StandardScaler(), ["amount", "oldbalanceOrg"])
    ]
)

print("\nPreprocessing pipeline created successfully!")

# Creating Logistic Regression Pipeline


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)



# Training Model

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Model trained successfully!")



# Making Predictions

y_pred = model.predict(X_test)


# Evaluating Model

accuracy = accuracy_score(y_test, y_pred)

print("\n LOGISTIC MODEL RESULTs:")
print("Accuracy:", accuracy)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Decision Tree Model
decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)

# Training data with decision tree model
decision_tree.fit(X_train, y_train)

# Predicting results
y_pred_tree = decision_tree.predict(X_test)

# Evaluating the decision tree results
print("\nDECISION TREE RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("F1 Score:", f1_score(y_test, y_pred_tree))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))


# Random Forest Model
random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Training data on random forest model
print("\nTraining Random Forest...")
random_forest.fit(X_train, y_train)

# Predicting results
y_pred_rf = random_forest.predict(X_test)

# Evaluating the random forest results
print("\nRANDOM FOREST RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


"""After evaluating all three models and comparing the results the random forest shows better precision and f1 score 
    so, considering the Random Forest Model for apps main algorithm"""

# Save the Random Forest model
joblib.dump(random_forest, "src/fraud_model.pkl")

print("\nRandom Forest model saved successfully!")