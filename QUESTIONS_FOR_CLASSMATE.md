# Questions to Ask Classmate About Their 90% Score

**Purpose:** Understand how they achieved 90% with "Random Forest + Grid Search"  
**Our Result:** 78.26% with GridSearchCV RF (11.74pp gap)  
**Time:** 5-10 minutes to ask, 30 minutes to analyze answers

---

## 🎯 CRITICAL QUESTIONS (Ask These First)

### 1. Competition Verification
**Question:** "What's the exact Kaggle competition name and link? Can you share it?"

**Why:** Verify it's the same competition (most likely explanation for 11.74pp gap)

**Follow-up:** "What's the train/test split size? (Ours is 646 train, 277 test)"

---

### 2. Proof of Score
**Question:** "Can you share a screenshot of your 90% Kaggle submission? Or your Kaggle username so I can verify?"

**Why:** Confirm the score is real and not a misunderstanding

---

### 3. Cross-Validation Score
**Question:** "What was your cross-validation accuracy before submitting?"

**Why:** 
- If CV >85%: Possible data leakage
- If CV 78-82%: Similar to ours, large positive gap is suspicious
- If CV <78%: Inconsistent with 90% Kaggle

**Red Flag:** CV >85% suggests data leakage

---

## 📊 FEATURE QUESTIONS

### 4. Feature Count
**Question:** "How many features did you use in your final model? (We used 31 original features)"

**Why:** Reveals if they did feature engineering

**Follow-up:** "Did you create any new features, or just use the original ones?"

---

### 5. Feature Engineering Details
**Question:** "If you created new features, which ones? Can you share the top 5 most important features?"

**Why:** Specific features we can try

**Follow-up:** "Did you use any external data sources (company databases, funding data, etc.)?"

---

### 6. Feature Selection
**Question:** "Did you do any feature selection? If so, what method?"

**Why:** Maybe they removed noisy features we kept

---

## 🔧 PREPROCESSING QUESTIONS

### 7. Missing Value Handling
**Question:** "How did you handle missing values? (We used median for numeric, mode for categorical)"

**Why:** Different imputation might help

---

### 8. Scaling/Normalization
**Question:** "Did you scale or normalize features? If so, which method?"

**Why:** Random Forest doesn't need scaling, but maybe they did something special

---

### 9. Categorical Encoding
**Question:** "How did you encode categorical variables? (We used one-hot encoding)"

**Why:** Different encoding might help

---

## 🤖 MODEL QUESTIONS

### 10. Grid Search Parameters
**Question:** "What parameter ranges did you test in your Grid Search? Can you share your param_grid?"

**Why:** Maybe they tested ranges we didn't

**Our ranges:**
```python
{
  'n_estimators': [400, 500, 600],
  'max_depth': [8, 10, 12],
  'min_samples_split': [2, 5, 8],
  'min_samples_leaf': [1, 2],
  'max_features': ['log2', 0.3],
  'class_weight': [None, 'balanced']
}
```

---

### 11. Best Parameters
**Question:** "What were your best parameters from Grid Search?"

**Why:** Compare to ours to see what's different

**Our best:**
```python
{
  'n_estimators': 600,
  'max_depth': 12,
  'min_samples_split': 2,
  'min_samples_leaf': 1,
  'max_features': 'log2',
  'class_weight': None
}
```

---

### 12. Cross-Validation Strategy
**Question:** "What cross-validation strategy did you use? (We used 10-fold Stratified K-Fold)"

**Why:** Different CV might reveal different optimal parameters

---

### 13. Single Model or Ensemble
**Question:** "Did you use a single Random Forest, or did you ensemble multiple models?"

**Why:** "Random Forest + Grid Search" sounds like single model, but maybe they ensembled

**Follow-up:** "If ensemble, how many models and how did you combine them?"

---

## 🚨 DATA LEAKAGE QUESTIONS

### 14. Validation Approach
**Question:** "How did you validate your model to avoid data leakage?"

**Why:** Check if they accidentally used test data

---

### 15. Test Data Usage
**Question:** "Did you ever look at or use the test data during model development?"

**Why:** Direct check for leakage

**Red Flag:** If they say "yes" to any test data usage

---

### 16. Feature Creation Timing
**Question:** "When did you create features - before or after splitting train/test?"

**Why:** Creating features after seeing test data is leakage

**Red Flag:** If they created features after seeing test data

---

## 📋 SUMMARY QUESTIONS

### 17. Overall Approach
**Question:** "Can you describe your overall approach in 3-5 steps? Like: 1) Load data, 2) Create features, 3) Grid Search, etc."

**Why:** Get high-level understanding of their workflow

---

### 18. Key Success Factor
**Question:** "What do you think was the key factor that got you to 90%? What made the biggest difference?"

**Why:** They might reveal the secret directly

---

### 19. Reproducibility
**Question:** "If I follow your exact approach, should I be able to get 90% too? Or was there some luck involved?"

**Why:** Understand if it's reproducible

---

### 20. Willingness to Share
**Question:** "Would you be willing to share your code or notebook? Or at least the key parts?"

**Why:** Best way to understand their approach

---

## 🎯 How to Analyze Answers

### Red Flags (STOP if you see these)

**Different Competition:**
- Different train/test size
- Different competition name
- Different evaluation metric
- **Action:** STOP - not comparable

**Data Leakage:**
- CV accuracy >85%
- Used test data in any way
- Created features after seeing test data
- Small CV-to-Kaggle gap with high score
- **Action:** STOP - invalid score

**Vague/Inconsistent Answers:**
- Can't remember details
- Inconsistent numbers
- Won't share specifics
- **Action:** STOP - likely not reproducible

---

### Green Flags (Try their approach if you see these)

**Same Competition:**
- Same train/test size (646/277)
- Same competition name
- Can show screenshot
- **Action:** Continue investigating

**Specific Reproducible Details:**
- Shares exact parameters
- Shares specific features created
- Shares preprocessing steps
- Consistent CV and Kaggle scores
- **Action:** Try their approach

**Novel Technique:**
- Feature engineering we didn't try
- Preprocessing we didn't try
- Parameter ranges we didn't test
- **Action:** Implement their technique

---

## ✅ What to Do After Getting Answers

### Scenario 1: Different Competition
- ✅ STOP immediately
- ✅ Accept 81.88% as excellent for our competition
- ✅ Not comparable, not useful

### Scenario 2: Data Leakage Detected
- ✅ STOP immediately
- ✅ Their score is invalid
- ✅ Accept 81.88% as legitimate

### Scenario 3: Vague/Won't Share
- ✅ STOP - likely not reproducible
- ✅ Accept 81.88% as our best
- ✅ Move on

### Scenario 4: Specific Reproducible Approach
- ✅ Implement their exact approach
- ✅ Test on our dataset
- ✅ If works: Great! If not: Accept 81.88%

---

## 📝 Template Message to Classmate

**Subject:** Question about your 90% Kaggle score

**Message:**
```
Hi [Name],

Congrats on your 90% Kaggle score! I'm trying to improve my model 
(currently at 81.88%) and you mentioned you used Random Forest + Grid Search.

I also used GridSearchCV with Random Forest but only got 78.26%. 
I'm trying to understand what I might be missing.

Could you help me with a few quick questions?

1. What's the exact Kaggle competition link? (Want to make sure we're 
   doing the same one - mine has 646 train, 277 test samples)

2. What was your cross-validation accuracy before submitting?

3. How many features did you use? Did you create any new features?

4. What parameter ranges did you test in Grid Search?

5. What were your best parameters?

Would you be willing to share your approach or code? Or at least the 
key steps that made the biggest difference?

Thanks!
[Your Name]
```

---

## ⏰ Timeline

**Step 1:** Ask questions (5-10 minutes)  
**Step 2:** Wait for answers (varies)  
**Step 3:** Analyze answers (20-30 minutes)  
**Step 4:** Decide next action (5 minutes)  
**Total:** 30-45 minutes + waiting time

---

**Status:** ⏸️ **READY TO ASK CLASSMATE**  
**Next Step:** Send questions and wait for answers  
**Then:** Analyze and decide based on responses


