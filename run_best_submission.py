"""
Generate best submission using extensive RF tuning with feature engineering.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name, save_submission
from src.features import split_columns, build_preprocessor
from src.modeling import build_pipelines
from src.feature_engineering import engineer_all_features
from src.evaluation import evaluate_all

print("\n" + "="*70)
print("GENERATING BEST SUBMISSION")
print("="*70)
print("\nStrategy: Extensive RF + Feature Engineering")
print("="*70 + "\n")

# Configuration
DATA_DIR = 'data'
SEED = 42
np.random.seed(SEED)

# Load data
print("1. Loading data...")
train_df, test_df, sample_submission_df = load_data(DATA_DIR)
target_name = get_target_name(sample_submission_df)

X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
y_train = train_df[target_name]
X_test = test_df.drop(columns=['id'], errors='ignore')

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# Feature engineering
print("\n2. Applying feature engineering...")
X_train_eng = engineer_all_features(X_train, y=y_train, create_poly=True, select_features=False)
X_test_eng = engineer_all_features(X_test, y=None, create_poly=True, select_features=False)

print(f"✓ Features: {X_train.shape[1]} → {X_train_eng.shape[1]}")

# Build preprocessor
print("\n3. Building preprocessor...")
numeric_cols, categorical_cols = split_columns(X_train_eng)
preprocessor = build_preprocessor(numeric_cols, categorical_cols)

# Build RF pipeline
print("\n4. Building Random Forest pipeline...")
pipelines = build_pipelines(preprocessor)
rf_pipeline = pipelines['rf']

# Load best parameters
print("\n5. Loading best hyperparameters...")
best_params_path = "reports/best_rf_extensive_params.json"

if os.path.exists(best_params_path):
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    print("✓ Best parameters loaded:")
    for param, value in best_params.items():
        param_name = param.replace('clf__', '')
        setattr(rf_pipeline.named_steps['clf'], param_name, value)
        print(f"  - {param_name}: {value}")
else:
    print("⚠ Warning: Best parameters not found. Using default parameters.")

# Cross-validation
print("\n6. Running cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(rf_pipeline, X_train_eng, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Fold scores: {[f'{score:.4f}' for score in cv_scores]}")

# Train on full training set
print("\n7. Training on full training set...")
rf_pipeline.fit(X_train_eng, y_train)
print("✓ Training complete!")

# Generate predictions
print("\n8. Generating predictions...")
predictions = rf_pipeline.predict(X_test_eng)

# Create submission
submission_df = sample_submission_df.copy()
submission_df[target_name] = predictions

# Save submission
submission_path = 'submission_advanced.csv'
save_submission(submission_df, submission_path)

print(f"\n✓ Submission saved to: {submission_path}")
print(f"  - Predictions: {len(predictions)}")
print(f"  - Class distribution: {pd.Series(predictions).value_counts().to_dict()}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n📊 Model: Random Forest (Extensive Tuning + Feature Engineering)")
print(f"📈 CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"📁 Submission: {submission_path}")
print(f"\n🎯 Expected Improvement:")
print(f"  - Previous Kaggle: 78.26%")
print(f"  - Current CV: {cv_scores.mean()*100:.2f}%")
print(f"  - Difference: {(cv_scores.mean() - 0.7826)*100:+.2f}%")

if cv_scores.mean() >= 0.80:
    print(f"\n✅ TARGET MET: CV score ≥ 80%")
else:
    print(f"\n⚠️  TARGET NOT MET: CV score < 80%")
    print(f"   Gap to target: {(0.80 - cv_scores.mean())*100:.2f}%")

print("\n🚀 Next Steps:")
print(f"  1. Upload {submission_path} to Kaggle")
print(f"  2. Check leaderboard score")
print(f"  3. If still < 80%, try:")
print(f"     - More aggressive feature selection")
print(f"     - Different model architectures")
print(f"     - Ensemble methods")

print("\n" + "="*70)
print("✓ COMPLETE!")
print("="*70 + "\n")

