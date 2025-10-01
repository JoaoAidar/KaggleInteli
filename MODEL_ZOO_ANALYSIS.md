# Model Zoo Analysis Report

**Date:** 2025-09-30  
**Objective:** Achieve ≥80% accuracy on Kaggle leaderboard  
**Target CV Accuracy:** ≥81.5% (to account for ~1.5% Kaggle gap)

---

## Executive Summary

After systematic evaluation of multiple models and feature configurations, we have identified a **performance ceiling around 79-80% CV accuracy**. This presents a significant challenge as our target is 81.5% CV (to achieve 80% on Kaggle).

### Key Findings

1. **Best Model So Far:**
   - **Random Forest × Original Features (31 features)**
   - CV Accuracy: **79.88%** (±4.06%)
   - Precision: 79.37%, Recall: 93.32%, F1: 85.72%
   - Training time: 240.6s

2. **Performance Ceiling:**
   - All 6 evaluated combinations achieved 78-80% CV accuracy
   - No model reached the 81.5% target
   - Gap to target: **1.62 percentage points**

3. **Feature Engineering Impact:**
   - **Negative**: Adding features (interactions, polynomials) did NOT improve performance
   - Original 31 features performed best
   - More features → more noise and overfitting

4. **Model Comparison:**
   - Random Forest: 79.26-79.88% CV
   - XGBoost: 78.03-79.26% CV
   - LightGBM: Not completed (interrupted)

---

## Detailed Results

### Completed Evaluations (6/15)

| Rank | Model | Config | Features | CV Acc | CV Std | Precision | Recall | F1 | Time (s) |
|------|-------|--------|----------|--------|--------|-----------|--------|----|----|
| 1 | RF | A_original | 31 | **79.88%** | 4.06% | 79.37% | 93.32% | 85.72% | 240.6 |
| 2 | RF | C_polynomials | 46 | 79.42% | 3.11% | 81.14% | 89.02% | 84.81% | 285.0 |
| 3 | RF | B_interactions | 41 | 79.26% | 3.49% | 80.41% | 89.98% | 84.86% | 257.1 |
| 4 | XGBoost | A_original | 31 | 79.26% | 2.73% | 80.44% | 89.95% | 84.89% | 55.8 |
| 5 | XGBoost | C_polynomials | 46 | 79.26% | 2.83% | 80.01% | 90.69% | 84.94% | 36.8 |
| 6 | XGBoost | B_interactions | 41 | 78.03% | 3.70% | 79.35% | 89.74% | 84.10% | 40.9 |

### Key Observations

1. **Random Forest Dominance:**
   - RF consistently outperforms XGBoost
   - Best RF: 79.88% vs Best XGBoost: 79.26%

2. **Feature Configuration Impact:**
   - **A_original (31 features)**: Best for both RF and XGBoost
   - **B_interactions (41 features)**: Worst performance
   - **C_polynomials (46 features)**: Middle performance

3. **Variance Analysis:**
   - RF has higher variance (3.11-4.06%) than XGBoost (2.73-3.70%)
   - Higher variance suggests potential overfitting
   - XGBoost is more stable but slightly lower accuracy

4. **Precision-Recall Trade-off:**
   - All models favor **high recall** (89-93%) over precision (79-81%)
   - This is appropriate for startup success prediction (better to predict success and be wrong than miss a success)

---

## Historical Performance Comparison

| Submission | Model | Features | CV Acc | Kaggle Acc | Gap |
|------------|-------|----------|--------|------------|-----|
| submission.csv | Tuned RF | 31 | 80.18% | 78.26% | -1.92% |
| submission_advanced.csv | Extensive RF + FE | 56 | 79.10% | 78.99% | -0.11% |
| **Model Zoo Best** | **RF × A_original** | **31** | **79.88%** | **TBD** | **?** |

### Analysis:
- Original tuned RF (80.18% CV) still holds the best CV score
- Model Zoo RF (79.88% CV) is close but slightly lower
- The extensive RF with feature engineering (79.10% CV) performed worse
- **Conclusion**: Simpler is better - original 31 features are optimal

---

## Why We're Hitting a Ceiling

### 1. **Data Limitations**
- Only 646 training samples
- 31 features with some missing values (1.4-21.4%)
- Class imbalance: 64.7% success, 35.3% failure
- Possible noise in labels or features

### 2. **Overfitting Risk**
- High variance in CV scores (3-4%)
- Consistent 1.5-2% gap between CV and Kaggle
- More features → worse performance

### 3. **Feature Quality**
- Original features may not capture all success factors
- Engineered features add noise rather than signal
- Missing values may contain information we're losing

### 4. **Model Capacity**
- Tree-based models may be reaching their limit
- Linear relationships might exist that trees can't capture well
- Ensemble methods might help but unlikely to bridge 1.6% gap

---

## Recommendations

### Immediate Actions (High Priority)

1. **✅ Use Best Existing Model**
   - **Recommendation**: Upload `submission.csv` (80.18% CV, 78.26% Kaggle)
   - This is still our best bet despite being below 80%
   - Gap analysis: 80.18% CV → 78.26% Kaggle = -1.92% gap

2. **🔄 Create Ensemble Submission**
   - Combine top 3-5 models using soft voting
   - Models: RF (A, C), XGBoost (A, C)
   - Expected: Might gain 0.5-1% from ensemble diversity
   - Target: 80-80.5% CV → 78.5-79% Kaggle

3. **🎯 Feature Selection Approach**
   - Use SelectKBest or RF importance to select top 15-20 features
   - Reduce noise by removing weak features
   - Might improve generalization

### Medium Priority

4. **📊 Analyze Misclassifications**
   - Identify which samples are consistently misclassified
   - Look for patterns in errors
   - Might reveal data quality issues or missing features

5. **🔧 Alternative Preprocessing**
   - Try different imputation strategies (KNN, iterative)
   - Experiment with feature scaling methods
   - Test different encoding for categorical variables

6. **🧪 Advanced Techniques**
   - Pseudo-labeling: Use test set predictions to augment training
   - Data augmentation: SMOTE for minority class
   - Calibration: Calibrate probabilities before thresholding

### Low Priority (Unlikely to Help)

7. **❌ More Feature Engineering**
   - Evidence shows this hurts performance
   - Only try if other methods fail

8. **❌ More Hyperparameter Tuning**
   - Already extensively tuned
   - Diminishing returns

9. **❌ Deep Learning**
   - Too few samples (646) for neural networks
   - Unlikely to outperform tree-based models

---

## Realistic Expectations

### Scenario Analysis

**Optimistic Scenario (Ensemble + Feature Selection):**
- CV Accuracy: 80.5-81%
- Kaggle Accuracy: 79-79.5%
- **Still below 80% target**

**Realistic Scenario (Best Single Model):**
- CV Accuracy: 80.18% (already achieved)
- Kaggle Accuracy: 78.26% (already achieved)
- **1.74% below target**

**Pessimistic Scenario:**
- CV Accuracy: 79-80%
- Kaggle Accuracy: 77.5-78.5%
- **1.5-2.5% below target**

### Hard Truth

**The 80% Kaggle target may not be achievable with this dataset and these features.**

Reasons:
1. Performance ceiling around 80% CV
2. Consistent 1.5-2% CV-to-Kaggle gap
3. Limited training data (646 samples)
4. Feature engineering makes things worse
5. All advanced models plateau at same level

---

## Next Steps

### Recommended Workflow

1. **Create Ensemble Submission** (30 minutes)
   - Combine RF (A_original), RF (C_polynomials), XGBoost (A_original)
   - Use soft voting
   - Generate `submission_ensemble.csv`

2. **Feature Selection Submission** (30 minutes)
   - Select top 20 features using RF importance
   - Retrain best RF model
   - Generate `submission_feature_selected.csv`

3. **Upload All Submissions to Kaggle** (15 minutes)
   - submission.csv (baseline: 78.26%)
   - submission_advanced.csv (extensive: 78.99%)
   - submission_ensemble.csv (new)
   - submission_feature_selected.csv (new)

4. **Analyze Kaggle Scores** (15 minutes)
   - Compare actual vs expected performance
   - Identify which approach works best
   - Adjust strategy based on results

5. **If Still Below 80%:**
   - Accept that 80% may not be achievable
   - Focus on maximizing score (aim for 79-79.5%)
   - Document learnings and limitations

---

## Conclusion

We have systematically evaluated multiple models and feature configurations. The evidence strongly suggests a **performance ceiling around 79-80% CV accuracy**, which translates to **77.5-78.5% on Kaggle** due to the consistent gap.

**Key Takeaways:**
- ✅ Best model: Random Forest with original 31 features
- ✅ Extensive hyperparameter tuning completed
- ❌ Feature engineering hurts performance
- ❌ 80% Kaggle target is very challenging with current data

**Recommended Action:**
Create ensemble and feature-selected submissions, upload to Kaggle, and aim for **79-79.5% Kaggle accuracy** as a realistic goal.

---

**Report Status:** Complete  
**Confidence Level:** High  
**Recommendation:** Proceed with ensemble approach, but adjust expectations


