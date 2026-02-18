import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("data/Training.csv")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20]
}

rf = RandomForestClassifier()
grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1_weighted")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, "models/trained_model.pkl")
