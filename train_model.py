"""
Iris Flower Classification - Model Training Script
----------------------------------------------------
Loads the Iris dataset, does a bit of EDA, trains a few classifiers,
picks the best one, evaluates it, and saves the trained model + scaler
to disk so the Streamlit frontend can load them instantly.

Run:  python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just saving PNGs
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("Iris.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Drop the Id column, it's not a feature
df = df.drop(columns=["Id"])

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"

# ---------------------------------------------------------------
# 2. QUICK EDA (saved as PNG files so you can look at them)
# ---------------------------------------------------------------
sns.pairplot(df, hue=TARGET, corner=True)
plt.savefig("eda_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df[FEATURES].corr(), annot=True, cmap="RdPu")
plt.title("Feature Correlation Heatmap")
plt.savefig("eda_correlation.png", dpi=120, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flatten(), FEATURES):
    sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax)
plt.tight_layout()
plt.savefig("eda_histograms.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved EDA plots: eda_pairplot.png, eda_correlation.png, eda_histograms.png")

# ---------------------------------------------------------------
# 3. PREPROCESS
# ---------------------------------------------------------------
X = df[FEATURES].values
y_raw = df[TARGET].values

le = LabelEncoder()
y = le.fit_transform(y_raw)  # Setosa=0, Versicolor=1, Virginica=2 (alphabetical)
print("Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. TRAIN MULTIPLE MODELS AND COMPARE
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=4),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")
    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "preds": preds,
    }
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

# ---------------------------------------------------------------
# 5. PICK THE BEST MODEL
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

# Detailed report + confusion matrix for the best model
best_preds = results[best_name]["preds"]
print("\nClassification Report:\n", classification_report(y_test, best_preds, target_names=le.classes_))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. SAVE MODEL, SCALER, LABEL ENCODER + METRICS FOR THE FRONTEND
# ---------------------------------------------------------------
joblib.dump(best_model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
joblib.dump(le, "iris_label_encoder.pkl")

metrics_summary = {
    "best_model_name": best_name,
    "all_results": {
        name: {k: v for k, v in r.items() if k != "model" and k != "preds"}
        for name, r in results.items()
    },
    "feature_names": FEATURES,
    "classes": list(le.classes_),
}
joblib.dump(metrics_summary, "iris_metrics.pkl")

print("\nSaved: iris_model.pkl, iris_scaler.pkl, iris_label_encoder.pkl, iris_metrics.pkl")
print("Done! Now run: streamlit run app.py")
