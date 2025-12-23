import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Check if model exists, if not train it first
model_path = 'fraud_model.pkl'
if not os.path.exists(model_path):
    print("Model not found. Training new model...")
    from fraud_model import train_and_evaluate_model
    model = train_and_evaluate_model('data/transactions.csv')
else:
    # Load the trained model
    from fraud_model import FraudDetectionModel
    model = FraudDetectionModel()
    model.load_model(model_path)

# Load dataset
df = pd.read_csv('data/transactions.csv')

# Use the correct target column name
y = df['is_fraud']

# Use the model's feature extraction method to ensure consistency with training
X = model.extract_features(df)

# Use only the features that the model was trained on
feature_cols = model.get_feature_columns()
available_cols = [col for col in feature_cols if col in X.columns]
X = X[available_cols]

print(f"Using features: {available_cols}")

# Split train/test with stratification to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict using the trained model
predictions, probabilities = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
cm = confusion_matrix(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:\n", cm)
