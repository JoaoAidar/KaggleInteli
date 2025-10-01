"""
Weighted Ensemble Using Actual Kaggle Scores
Phase 1, Task 3: Create weighted ensemble based on actual Kaggle performance

Target: +0.3-0.8pp improvement over 81.88%
Expected: 82.2-82.7% Kaggle
"""

import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from itertools import combinations

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("WEIGHTED ENSEMBLE - KAGGLE SCORE BASED")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Objective: Create weighted ensemble using actual Kaggle scores")
print("Current best: 81.88% Kaggle (majority_vote)")
print("Target: 82.2-82.7% Kaggle (+0.3-0.8pp improvement)")
print()

# ============================================================================
# LOAD EXISTING SUBMISSIONS
# ============================================================================

print("=" * 80)
print("LOADING EXISTING SUBMISSIONS")
print("=" * 80)
print()

# Define submissions with their Kaggle scores
submissions = {
    'majority_vote': {
        'file': 'submission_majority_vote.csv',
        'kaggle_score': 0.8188,
        'description': 'Hard voting ensemble (RF + RF + XGBoost)'
    },
    'voting_ensemble': {
        'file': 'submission_voting_ensemble.csv',
        'kaggle_score': 0.7971,
        'description': 'Soft voting ensemble'
    },
    'weighted_ensemble': {
        'file': 'submission_weighted_ensemble.csv',
        'kaggle_score': 0.7971,
        'description': 'Weighted voting ensemble'
    },
    'advanced': {
        'file': 'submission_advanced.csv',
        'kaggle_score': 0.7899,
        'description': 'Advanced RF with feature engineering'
    },
    'baseline': {
        'file': 'submission.csv',
        'kaggle_score': 0.7826,
        'description': 'Baseline RF'
    }
}

# Load all submissions
loaded_submissions = {}
for name, info in submissions.items():
    try:
        df = pd.read_csv(info['file'])
        loaded_submissions[name] = {
            'predictions': df['labels'].values,
            'kaggle_score': info['kaggle_score'],
            'description': info['description']
        }
        print(f"✓ Loaded {name}: {info['kaggle_score']:.4f} ({len(df)} predictions)")
    except FileNotFoundError:
        print(f"⚠️  Skipped {name}: File not found")

print()
print(f"Total submissions loaded: {len(loaded_submissions)}")
print()

# ============================================================================
# WEIGHTED ENSEMBLE STRATEGIES
# ============================================================================

print("=" * 80)
print("TESTING WEIGHTED ENSEMBLE STRATEGIES")
print("=" * 80)
print()

# Get test IDs from one of the submissions
sample_df = pd.read_csv(submissions['majority_vote']['file'])
test_ids = sample_df['id'].values

# Strategy 1: Kaggle Score Weights (Normalized)
print("Strategy 1: Kaggle Score Weights (Normalized)")
print("-" * 60)

weights_kaggle = np.array([info['kaggle_score'] for info in loaded_submissions.values()])
weights_kaggle_norm = weights_kaggle / weights_kaggle.sum()

print("Weights:")
for i, (name, info) in enumerate(loaded_submissions.items()):
    print(f"  {name}: {weights_kaggle_norm[i]:.4f} (Kaggle: {info['kaggle_score']:.4f})")
print()

# Create weighted predictions
predictions_array = np.array([info['predictions'] for info in loaded_submissions.values()])
weighted_pred_kaggle = np.average(predictions_array, axis=0, weights=weights_kaggle_norm)
final_pred_kaggle = (weighted_pred_kaggle >= 0.5).astype(int)

n_success = (final_pred_kaggle == 1).sum()
print(f"Predictions: {n_success} success ({n_success/len(final_pred_kaggle)*100:.1f}%), "
      f"{len(final_pred_kaggle)-n_success} failure ({(len(final_pred_kaggle)-n_success)/len(final_pred_kaggle)*100:.1f}%)")
print()

# Strategy 2: Squared Kaggle Score Weights (Emphasize best models)
print("Strategy 2: Squared Kaggle Score Weights (Emphasize best)")
print("-" * 60)

weights_squared = weights_kaggle ** 2
weights_squared_norm = weights_squared / weights_squared.sum()

print("Weights:")
for i, (name, info) in enumerate(loaded_submissions.items()):
    print(f"  {name}: {weights_squared_norm[i]:.4f} (Kaggle²: {info['kaggle_score']**2:.4f})")
print()

weighted_pred_squared = np.average(predictions_array, axis=0, weights=weights_squared_norm)
final_pred_squared = (weighted_pred_squared >= 0.5).astype(int)

n_success = (final_pred_squared == 1).sum()
print(f"Predictions: {n_success} success ({n_success/len(final_pred_squared)*100:.1f}%), "
      f"{len(final_pred_squared)-n_success} failure ({(len(final_pred_squared)-n_success)/len(final_pred_squared)*100:.1f}%)")
print()

# Strategy 3: Top 3 Models Only (Highest Kaggle scores)
print("Strategy 3: Top 3 Models Only")
print("-" * 60)

# Sort by Kaggle score
sorted_submissions = sorted(loaded_submissions.items(), 
                           key=lambda x: x[1]['kaggle_score'], 
                           reverse=True)
top3_names = [name for name, _ in sorted_submissions[:3]]
top3_weights = np.array([loaded_submissions[name]['kaggle_score'] for name in top3_names])
top3_weights_norm = top3_weights / top3_weights.sum()

print("Top 3 models:")
for i, name in enumerate(top3_names):
    info = loaded_submissions[name]
    print(f"  {name}: {top3_weights_norm[i]:.4f} (Kaggle: {info['kaggle_score']:.4f})")
print()

top3_predictions = np.array([loaded_submissions[name]['predictions'] for name in top3_names])
weighted_pred_top3 = np.average(top3_predictions, axis=0, weights=top3_weights_norm)
final_pred_top3 = (weighted_pred_top3 >= 0.5).astype(int)

n_success = (final_pred_top3 == 1).sum()
print(f"Predictions: {n_success} success ({n_success/len(final_pred_top3)*100:.1f}%), "
      f"{len(final_pred_top3)-n_success} failure ({(len(final_pred_top3)-n_success)/len(final_pred_top3)*100:.1f}%)")
print()

# Strategy 4: Exponential Weights (Heavily favor best)
print("Strategy 4: Exponential Weights (Heavily favor best)")
print("-" * 60)

weights_exp = np.exp(weights_kaggle * 5)  # Scale factor of 5
weights_exp_norm = weights_exp / weights_exp.sum()

print("Weights:")
for i, (name, info) in enumerate(loaded_submissions.items()):
    print(f"  {name}: {weights_exp_norm[i]:.4f} (exp(Kaggle×5): {weights_exp[i]:.2f})")
print()

weighted_pred_exp = np.average(predictions_array, axis=0, weights=weights_exp_norm)
final_pred_exp = (weighted_pred_exp >= 0.5).astype(int)

n_success = (final_pred_exp == 1).sum()
print(f"Predictions: {n_success} success ({n_success/len(final_pred_exp)*100:.1f}%), "
      f"{len(final_pred_exp)-n_success} failure ({(len(final_pred_exp)-n_success)/len(final_pred_exp)*100:.1f}%)")
print()

# ============================================================================
# SELECT BEST STRATEGY
# ============================================================================

print("=" * 80)
print("STRATEGY SELECTION")
print("=" * 80)
print()

strategies = {
    'kaggle_weights': final_pred_kaggle,
    'squared_weights': final_pred_squared,
    'top3_only': final_pred_top3,
    'exponential_weights': final_pred_exp
}

# Compare strategies
print("Strategy Comparison:")
print("-" * 60)
for name, pred in strategies.items():
    n_success = (pred == 1).sum()
    pct_success = n_success / len(pred) * 100
    
    # Check if different from majority_vote
    majority_vote_pred = loaded_submissions['majority_vote']['predictions']
    n_different = (pred != majority_vote_pred).sum()
    pct_different = n_different / len(pred) * 100
    
    print(f"{name}:")
    print(f"  Success: {n_success} ({pct_success:.1f}%)")
    print(f"  Different from majority_vote: {n_different} ({pct_different:.1f}%)")
    print()

# Select strategy with most difference from majority_vote (most potential for improvement)
differences = {name: (pred != loaded_submissions['majority_vote']['predictions']).sum() 
               for name, pred in strategies.items()}
best_strategy = max(differences, key=differences.get)

print(f"Selected strategy: {best_strategy}")
print(f"Reason: Most different from current best ({differences[best_strategy]} predictions)")
print()

final_predictions = strategies[best_strategy]

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("Saving submission file...")
submission_file = 'submission_weighted_kaggle.csv'

submission_df = pd.DataFrame({
    'id': test_ids,
    'labels': final_predictions
})
submission_df.to_csv(submission_file, index=False)

print(f"✓ Submission saved: {submission_file}")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("Saving results...")

n_success = (final_predictions == 1).sum()
n_failure = (final_predictions == 0).sum()

results_data = {
    'strategy': best_strategy,
    'models_used': list(loaded_submissions.keys()),
    'model_weights': {name: float(loaded_submissions[name]['kaggle_score']) 
                     for name in loaded_submissions.keys()},
    'prediction_distribution': {
        'success': int(n_success),
        'failure': int(n_failure),
        'pct_success': float(n_success / len(final_predictions) * 100),
        'pct_failure': float(n_failure / len(final_predictions) * 100)
    },
    'difference_from_majority_vote': int(differences[best_strategy]),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('reports/weighted_ensemble_kaggle_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print("✓ Results saved: reports/weighted_ensemble_kaggle_results.json")
print()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("WEIGHTED ENSEMBLE - SUMMARY")
print("=" * 80)
print()

print(f"Strategy: {best_strategy}")
print(f"Models: {len(loaded_submissions)}")
print(f"Predictions: {len(final_predictions)}")
print()

print(f"Submission File: {submission_file}")
print()

# Expected Kaggle performance
print("Expected Kaggle Performance:")
print("-" * 60)

# Estimate based on weighted average of Kaggle scores
weighted_avg_kaggle = np.average([info['kaggle_score'] for info in loaded_submissions.values()],
                                weights=weights_kaggle_norm)

print(f"  Weighted average of input scores: {weighted_avg_kaggle*100:.2f}%")
print(f"  Expected range: {(weighted_avg_kaggle-0.005)*100:.2f}% - {(weighted_avg_kaggle+0.005)*100:.2f}%")
print()

improvement = (weighted_avg_kaggle - 0.8188) * 100
print(f"Expected improvement over 81.88%: {improvement:+.2f}pp")
print()

if weighted_avg_kaggle >= 0.82:
    print("⚠️  May marginally improve to 82%+")
else:
    print("⚠️  Unlikely to improve significantly over 81.88%")

print()
print("Note: Weighted ensemble typically performs between best and average of inputs")
print("      Best input: 81.88%, Average: ~79.5%, Expected: ~80-82%")
print()

print("=" * 80)
print("NEXT STEP: Upload submission_weighted_kaggle.csv to Kaggle")
print("=" * 80)
print()

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

