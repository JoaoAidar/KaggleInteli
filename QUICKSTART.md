# Quick Start Guide

Get started with the Startup Success Prediction project in 5 minutes!

---

## 🚀 Fastest Path to Submission

### Option 1: One Command (Recommended)

```bash
make submit
```

This will:
1. Train the tuned Random Forest model
2. Generate `submission.csv`
3. Validate the submission format

**Time:** ~30 seconds

---

### Option 2: Complete Pipeline

```bash
make all
```

This will run:
1. `make eda` - Exploratory data analysis
2. `make cv` - Cross-validation (3 models)
3. `make tune` - Hyperparameter tuning (~5-10 minutes)
4. `make submit` - Generate submission

**Time:** ~10-15 minutes

---

## 📊 Step-by-Step Workflow

### 1. Explore the Data

```bash
make eda
```

**Output:**
- Dataset shapes
- Missing values
- Feature statistics
- Target distribution

**Time:** ~5 seconds

---

### 2. Evaluate Models

```bash
make cv
```

**Output:**
- Cross-validation results for 3 models
- Saved to `reports/cv_metrics.csv`

**Expected Results:**
- Random Forest: ~78-80% accuracy
- Gradient Boosting: ~78-80% accuracy
- Logistic Regression: ~75-77% accuracy

**Time:** ~30 seconds

---

### 3. Tune Hyperparameters (Optional but Recommended)

```bash
make tune
```

**Output:**
- Best Random Forest parameters
- Saved to `reports/best_rf_params.json`
- Expected improvement: +1-3% accuracy

**Time:** ~5-10 minutes

---

### 4. Generate Submission

```bash
# Option A: Use baseline Random Forest
make train

# Option B: Use tuned Random Forest (recommended)
make train-best
```

**Output:**
- `submission.csv` with 277 predictions
- Format validated automatically

**Time:** ~30 seconds

---

## 🎓 Interactive Analysis (Jupyter Notebook)

### Launch Notebook

```bash
jupyter notebook notebooks/01_startup_success.ipynb
```

### Run All Cells

1. Click **Cell** → **Run All**
2. Wait for all cells to execute (~2-3 minutes)
3. Review visualizations and results
4. Submission file will be generated automatically

### Notebook Sections

1. **Introduction** - Competition overview
2. **Data Loading** - Load datasets
3. **Data Cleaning** - Check for issues
4. **EDA** - Visualizations and insights
5. **Hypotheses** - Testable predictions
6. **Feature Engineering** - Preprocessing pipeline
7. **Model Building** - Three baseline models
8. **Cross-Validation** - Performance evaluation
9. **Hyperparameter Tuning** - Optimize Random Forest
10. **Final Training** - Generate predictions
11. **Conclusion** - Summary and next steps
12. **CLI Reproducibility** - Command-line examples

---

## 🔍 Verify Everything Works

```bash
python verify_project.py
```

**Expected Output:**
```
✅ ALL CHECKS PASSED!
Checks passed: 21/21 (100.0%)
```

---

## 📈 Check Your Results

### View Cross-Validation Results

```bash
# Windows
type reports\cv_metrics.csv

# Mac/Linux
cat reports/cv_metrics.csv
```

### View Submission

```bash
# Windows
type submission.csv | more

# Mac/Linux
head -20 submission.csv
```

### Check Submission Format

```bash
python -c "import pandas as pd; df = pd.read_csv('submission.csv'); print(f'Shape: {df.shape}'); print(f'Columns: {df.columns.tolist()}'); print(df.head())"
```

---

## 🎯 Model Selection Guide

### Which Model Should I Use?

| Model | Accuracy | Speed | Recommendation |
|-------|----------|-------|----------------|
| **Random Forest (tuned)** | ~80-82% | Medium | ⭐ **Best for submission** |
| Random Forest (baseline) | ~78-80% | Fast | Good for quick testing |
| Gradient Boosting | ~78-80% | Medium | Alternative option |
| Logistic Regression | ~75-77% | Very Fast | Baseline comparison |

### How to Use Each Model

```bash
# Random Forest (tuned) - RECOMMENDED
make tune
make train-best

# Random Forest (baseline)
python -m src.cli train-predict --model rf

# Gradient Boosting
python -m src.cli train-predict --model gb

# Logistic Regression
python -m src.cli train-predict --model logit
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'src'"

**Solution:** Make sure you're in the project root directory

```bash
cd e:\DOWNLOADS\KaggleInteli
python -m src.cli --help
```

---

### Issue: "FileNotFoundError: train.csv not found"

**Solution:** Verify data files exist

```bash
dir data\  # Windows
ls data/   # Mac/Linux
```

Expected files:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

---

### Issue: Makefile commands not working on Windows

**Solution:** Use Python CLI directly

```bash
# Instead of: make eda
python -m src.cli eda

# Instead of: make cv
python -m src.cli cv

# Instead of: make tune
python -m src.cli tune

# Instead of: make submit
python -m src.cli tune
python -m src.cli train-predict --use-best-rf
```

---

### Issue: Notebook kernel crashes during tuning

**Solution:** Reduce computational load

Edit `src/modeling.py` and change:
```python
# From:
n_iter=30

# To:
n_iter=10
```

---

## 📚 Additional Resources

### Documentation

- **README.md** - Comprehensive project documentation
- **PROJECT_SUMMARY.md** - Project status and results
- **This file (QUICKSTART.md)** - Quick start guide

### Code Documentation

All functions have docstrings. To view:

```python
from src.io_utils import load_data
help(load_data)
```

### CLI Help

```bash
# General help
python -m src.cli --help

# Command-specific help
python -m src.cli eda --help
python -m src.cli cv --help
python -m src.cli tune --help
python -m src.cli train-predict --help
```

---

## 🎉 Success Checklist

Before submitting to Kaggle:

- [ ] Ran `python verify_project.py` - all checks passed
- [ ] Ran `make cv` - models evaluated
- [ ] Ran `make tune` - hyperparameters optimized (optional but recommended)
- [ ] Ran `make submit` - submission.csv generated
- [ ] Verified submission format:
  - [ ] Has 277 rows (+ 1 header)
  - [ ] Has columns: `id`, `labels`
  - [ ] No missing values
- [ ] Reviewed notebook (optional)
- [ ] Ready to upload to Kaggle!

---

## 🏆 Expected Competition Performance

### Baseline (No Tuning)

- **Random Forest**: ~78-80% accuracy
- **Leaderboard Position**: Mid-tier

### With Tuning

- **Tuned Random Forest**: ~80-82% accuracy
- **Leaderboard Position**: Upper mid-tier to top-tier

### To Improve Further

1. **Feature Engineering**
   - Create interaction features
   - Engineer domain-specific ratios
   - Handle class imbalance

2. **Advanced Tuning**
   - Increase `n_iter` in RandomizedSearchCV
   - Try GridSearchCV for fine-tuning
   - Tune other models (Gradient Boosting)

3. **Ensemble Methods**
   - Combine predictions from multiple models
   - Use stacking or blending

---

## 💡 Pro Tips

1. **Always run tuning before final submission**
   ```bash
   make tune
   make train-best
   ```

2. **Check CV results to understand model performance**
   ```bash
   make cv
   type reports\cv_metrics.csv
   ```

3. **Use the notebook for exploration, CLI for production**
   - Notebook: Interactive analysis and visualization
   - CLI: Automated, reproducible pipeline

4. **Save your best parameters**
   - The `tune` command saves parameters to `reports/best_rf_params.json`
   - These are automatically loaded when using `--use-best-rf`

5. **Validate before submitting**
   ```bash
   python -c "import pandas as pd; sub = pd.read_csv('submission.csv'); test = pd.read_csv('data/test.csv'); assert len(sub) == len(test), 'Row count mismatch!'; print('✓ Validation passed!')"
   ```

---

## 🚀 Ready to Submit?

```bash
# Generate your best submission
make tune
make train-best

# Verify it's correct
python verify_project.py

# Upload submission.csv to Kaggle
# Good luck! 🎉
```

---

**Questions?** Check README.md or PROJECT_SUMMARY.md for more details!

