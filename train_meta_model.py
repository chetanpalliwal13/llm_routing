from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv("model_benchmark_dataset.csv")

# Convert categorical to numeric
df["Domain"] = df["Domain"].astype("category").cat.codes
df["Model"] = df["Model"].astype("category")
model_labels = df["Model"].cat.codes

# Features and labels
X = df[["Prompt Length", "Domain", "Latency", "RAM", "VRAM"]]
y = model_labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, clf.predict(X_test)))

# Save model for inference
joblib.dump(clf, "model_selector.pkl")
joblib.dump(df["Model"].cat.categories, "model_classes.pkl")
