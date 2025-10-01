# Startup Success Prediction - Kaggle Competition

**Competition:** [Inteli-M3] Campeonato 2025

A complete, production-ready machine learning pipeline for predicting startup success using scikit-learn.

---

## 📋 Project Overview

### Competition Objective

Predict startup success (binary classification) based on features including:
- Funding information (amounts, rounds, investors)
- Geographic location (state indicators)
- Industry category
- Milestone achievements
- Relationship networks

### Dataset Description

- **Training Set**: 647 startups with known outcomes
- **Test Set**: 278 startups requiring predictions
- **Features**: 32 columns (numeric and categorical)
- **Target**: Binary label (0 = failure, 1 = success)

### Target Metric

- **Primary**: Accuracy ≥ 80%
- **Secondary**: Precision, Recall, F1-score

---

## 🛠 Technical Stack

### Allowed Libraries

**Core ML/Data:**
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms

**Visualization:**
- `matplotlib` - Primary visualization (required)
- `seaborn` - Optional styling enhancements

**Other:**
- `jupyter` - Interactive notebook environment

### Constraints

✓ No external data sources (only `data/` directory)  
✓ All preprocessing in pipelines (no data leakage)  
✓ Fixed random seeds (`random_state=42`)  
✓ Python 3.8+ compatible

---

## 📁 Project Structure

```
.
├── data/                          # User-provided datasets
│   ├── train.csv                  # Training data with labels
│   ├── test.csv                   # Test data for predictions
│   └── sample_submission.csv      # Submission format template
├── notebooks/
│   └── 01_startup_success.ipynb   # Main analysis notebook (12 sections)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── io_utils.py                # Data loading/saving utilities
│   ├── features.py                # Feature engineering & preprocessing
│   ├── modeling.py                # Model building & hyperparameter tuning
│   ├── evaluation.py              # Metrics & cross-validation
│   └── cli.py                     # Command-line interface
├── reports/                       # Generated reports (created by pipeline)
│   ├── cv_metrics.csv             # Cross-validation results
│   └── best_rf_params.json        # Optimal RF hyperparameters
├── Makefile                       # Automation commands
├── README.md                      # This file
└── submission.csv                 # Final predictions (generated)
```

---

## 🚀 Setup Instructions

### Local Environment

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Verify data files exist
ls data/
# Expected: train.csv, test.csv, sample_submission.csv

# Verify project structure
python -c "from src.io_utils import load_data; print('✓ Setup complete!')"
```

### Kaggle Environment

1. Upload `notebooks/01_startup_success.ipynb` to Kaggle
2. Attach the competition dataset
3. Run all cells sequentially
4. Download `submission.csv`

---

## 💻 Usage

### Option 1: Command-Line Interface (Recommended)

#### Using Makefile (Simplest)

```bash
# Run exploratory data analysis
make eda

# Cross-validation evaluation (all models)
make cv

# Hyperparameter tuning (Random Forest)
make tune

# Generate submission with default RF
make train

# Generate submission with tuned RF (recommended)
make train-best

# Quick submission (runs train-best)
make submit

# Run complete pipeline: eda → cv → tune → submit
make all

# Clean generated files
make clean
```

#### Using Python CLI Directly

```bash
# Exploratory Data Analysis
python -m src.cli eda --data-dir data

# Cross-validation evaluation
python -m src.cli cv --data-dir data --output reports/cv_metrics.csv

# Hyperparameter tuning
python -m src.cli tune --data-dir data --seed 42 --output reports/best_rf_params.json

# Train and predict (default RF)
python -m src.cli train-predict --data-dir data --model rf --output submission.csv

# Train and predict (tuned RF)
python -m src.cli train-predict --data-dir data --use-best-rf --output submission.csv

# Train and predict (Gradient Boosting)
python -m src.cli train-predict --data-dir data --model gb --output submission.csv
```

### Option 2: Jupyter Notebook (Interactive)

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/01_startup_success.ipynb
# Run all cells sequentially (Cell → Run All)
# Submission file will be generated in project root
```

---

## 📊 Pipeline Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset shapes and info
- Missing value analysis
- Feature type identification
- Target distribution
- Correlation analysis

### 2. Feature Engineering
- **Numeric features**: Median imputation + StandardScaler
- **Categorical features**: Mode imputation + OneHotEncoder (min_frequency=10)
- All transformations in `ColumnTransformer` (no data leakage)

### 3. Model Building
- **Logistic Regression**: Fast baseline
- **Random Forest**: Ensemble method (primary model)
- **Gradient Boosting**: Alternative ensemble

### 4. Cross-Validation
- 5-fold Stratified K-Fold
- Metrics: Accuracy, Precision, Recall, F1-score
- Results saved to `reports/cv_metrics.csv`

### 5. Hyperparameter Tuning
- RandomizedSearchCV (30 iterations, 5-fold CV)
- Search space: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- Best parameters saved to `reports/best_rf_params.json`

### 6. Final Training & Prediction
- Train best model on 100% of training data
- Generate predictions for test set
- Create submission file matching required format

---

## ✅ Compliance Guarantees

| Requirement | Status | Details |
|------------|--------|---------|
| **Libraries** | ✓ | Only numpy, pandas, scikit-learn for ML |
| **Visualization** | ✓ | Only matplotlib (required) |
| **Data Sources** | ✓ | Only `data/` directory |
| **Data Leakage** | ✓ | All preprocessing in pipelines |
| **Reproducibility** | ✓ | Fixed `random_state=42` |
| **Submission Format** | ✓ | Matches `sample_submission.csv` exactly |

---

## 📈 Output Files

### Generated by Pipeline

| File | Description | Command |
|------|-------------|---------|
| `reports/cv_metrics.csv` | Cross-validation results for all models | `make cv` |
| `reports/best_rf_params.json` | Optimal Random Forest hyperparameters | `make tune` |
| `submission.csv` | Final predictions for Kaggle submission | `make submit` |

### Validation Checks

✓ `cv_metrics.csv` contains 3 rows (one per model)  
✓ `cv_metrics.csv` has columns: model, accuracy, precision, recall, f1  
✓ `submission.csv` has same columns as `sample_submission.csv`  
✓ `submission.csv` has same row count as `test.csv` (278 rows)  
✓ No missing values in submission  

---

## 🎯 Expected Results

### Model Performance (Typical)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.75-0.80 | ~0.70-0.78 | ~0.72-0.80 | ~0.71-0.79 |
| Random Forest | ~0.78-0.85 | ~0.75-0.83 | ~0.76-0.84 | ~0.75-0.83 |
| Gradient Boosting | ~0.76-0.82 | ~0.73-0.80 | ~0.74-0.81 | ~0.73-0.80 |

**Note:** Actual results depend on data characteristics and hyperparameter tuning.

### Threshold Achievement

- **Target**: ≥ 80% cross-validation accuracy
- **Expected**: Random Forest (tuned) typically meets or exceeds threshold

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`  
**Solution**: Run commands from project root directory

**Issue**: `FileNotFoundError: train.csv not found`  
**Solution**: Ensure data files are in `data/` directory

**Issue**: Notebook kernel crashes during tuning  
**Solution**: Reduce `n_iter` in `random_search_rf()` or use fewer CV folds

**Issue**: Makefile commands not working on Windows  
**Solution**: Use Python CLI directly or install `make` for Windows

---

## 📝 Development Notes

### Code Quality Standards

- **Style**: PEP 8 compliant
- **Docstrings**: All functions documented
- **Type Hints**: Used where helpful
- **Error Handling**: Graceful failures with clear messages

### Testing Recommendations

```bash
# Test data loading
python -c "from src.io_utils import load_data; load_data('data')"

# Test preprocessing
python -c "from src.features import split_columns, build_preprocessor; import pandas as pd; df = pd.DataFrame({'a': [1,2], 'b': ['x','y']}); print(split_columns(df))"

# Test CLI
python -m src.cli --help
```

---

## 🤝 Contributing

This project follows competition rules strictly. Suggested improvements:

1. **Feature Engineering**: Add interaction features, domain-specific ratios
2. **Model Exploration**: Try ensemble stacking, neural networks (if allowed)
3. **Hyperparameter Tuning**: Expand search space, use Bayesian optimization
4. **Validation**: Implement nested CV for unbiased estimates

---

## 📄 License

This project is created for educational purposes as part of the [Inteli-M3] Campeonato 2025 competition.

---

## 🙏 Acknowledgments

- **Competition Organizers**: [Inteli-M3] Campeonato 2025
- **Libraries**: scikit-learn, pandas, numpy, matplotlib
- **Community**: Kaggle community for inspiration and best practices

---

**Ready to compete? Run `make all` and submit your predictions!** 🚀

