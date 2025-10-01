"""
Test feature engineering module.
"""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.io_utils import load_data, get_target_name
from src.feature_engineering import engineer_all_features

print("="*70)
print("TESTING FEATURE ENGINEERING")
print("="*70)

# Load data
print("\n1. Loading data...")
train_df, test_df, sample_submission_df = load_data('data')
target_name = get_target_name(sample_submission_df)

X_train = train_df.drop(columns=[target_name, 'id'], errors='ignore')
y_train = train_df[target_name]

print(f"✓ Original features: {X_train.shape[1]}")

# Test feature engineering
print("\n2. Testing feature engineering...")
X_train_engineered = engineer_all_features(
    X_train, y=y_train,
    create_poly=True,
    select_features=False
)

print(f"\n✓ Engineered features: {X_train_engineered.shape[1]}")
print(f"✓ New features created: {X_train_engineered.shape[1] - X_train.shape[1]}")

# Show new feature names
new_features = [col for col in X_train_engineered.columns if col not in X_train.columns]
print(f"\nNew features ({len(new_features)}):")
for feat in new_features[:10]:  # Show first 10
    print(f"  - {feat}")
if len(new_features) > 10:
    print(f"  ... and {len(new_features) - 10} more")

print("\n" + "="*70)
print("✓ FEATURE ENGINEERING TEST PASSED!")
print("="*70)

