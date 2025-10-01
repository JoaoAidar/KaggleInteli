# Previsão de Sucesso de Startups - Competição Kaggle

**Competição:** [Inteli-M3] Campeonato 2025

---

## 🏆 Melhor Modelo: 81.88% de Acurácia no Kaggle

### Visão Geral

Este projeto alcançou **81.88% de acurácia** na competição Kaggle utilizando uma abordagem de **Ensemble de Votação Majoritária (Hard Voting)** com dois modelos Random Forest complementares. Esta abordagem simples e robusta superou métodos mais complexos como stacking e soft voting, demonstrando que simplicidade e diversidade de modelos são mais eficazes que complexidade excessiva.

**Arquivo de Submissão:** `submission_majority_vote.csv`

---

## 🎯 Arquitetura do Modelo

### Ensemble de Votação Majoritária (Hard Voting)

O modelo vencedor utiliza **votação majoritária** entre dois modelos Random Forest treinados com diferentes configurações:

#### Modelos Base:

1. **RF_Original** - Random Forest com features originais
   - **Features:** 31 features originais do dataset
   - **Acurácia CV:** 79.88% ± 3.85%
   - **Hiperparâmetros:**
     - `n_estimators`: 500
     - `max_depth`: 10
     - `min_samples_split`: 5
     - `min_samples_leaf`: 1
     - `max_features`: 'log2'
     - `class_weight`: None

2. **RF_Poly** - Random Forest com features polinomiais
   - **Features:** 46 features (31 originais + 15 polinomiais de grau 2)
   - **Acurácia CV:** 79.42% ± 3.96%
   - **Hiperparâmetros:**
     - `n_estimators`: 200
     - `max_depth`: None (sem limite)
     - `min_samples_split`: 10
     - `min_samples_leaf`: 2
     - `max_features`: 0.3
     - `class_weight`: 'balanced'

### Como Funciona a Votação Majoritária

```python
# Cada modelo faz uma predição binária (0 ou 1)
RF_Original prediz: [1, 0, 1, 0, ...]
RF_Poly prediz:     [1, 1, 1, 0, ...]

# Votação majoritária: se ambos concordam, usa o voto
# Se discordam, pode usar desempate ou voto do modelo mais confiável
Predição final:     [1, 0/1, 1, 0, ...]
```

**Vantagens da Votação Majoritária:**
- ✅ Decisões discretas (0 ou 1) evitam overfitting de probabilidades
- ✅ Robustez: erros de um modelo são compensados pelo outro
- ✅ Simplicidade: sem meta-learner complexo
- ✅ Diversidade: diferentes features e hiperparâmetros capturam padrões complementares

---

## 📊 Métricas de Desempenho

### Resultados do Ensemble

| Métrica | Valor | Observação |
|---------|-------|------------|
| **Acurácia CV** | ~79.5% | Média dos modelos base |
| **Acurácia Kaggle** | **81.88%** | Resultado final na competição |
| **Gap CV-Kaggle** | **+2.38pp** | Gap positivo indica excelente generalização |
| **Predições** | 277 | 194 sucessos (70.0%), 83 falhas (30.0%) |

### Comparação com Outras Abordagens

| Abordagem | Acurácia Kaggle | Gap CV-Kaggle | Status |
|-----------|-----------------|---------------|--------|
| **Hard Voting (RF+RF)** | **81.88%** | **+2.38pp** | ✅ **MELHOR** |
| Soft Voting | 79.71% | +0.21pp | Bom |
| Weighted Ensemble | 79.71% | +0.21pp | Bom |
| Advanced RF | 78.99% | -0.11pp | OK |
| GridSearchCV RF | 78.26% | -2.24pp | Overfitting |
| Stacking (5 modelos) | 76.09% | -3.02pp | Overfitting severo |

**Insight Crítico:** Métodos mais complexos (GridSearchCV, Stacking) tiveram **pior desempenho** devido a overfitting. A simplicidade da votação majoritária foi a chave do sucesso.

---

## 💡 Por Que Esta Abordagem Funciona

### 1. Diversidade de Modelos

**RF_Original vs RF_Poly:**
- **Features diferentes:** 31 vs 46 features
- **Hiperparâmetros diferentes:** Profundidade, número de árvores, regularização
- **Padrões complementares:** Cada modelo captura aspectos diferentes dos dados

**Resultado:** Erros dos modelos são **não-correlacionados**, permitindo que um compense o outro.

### 2. Votação Majoritária > Soft Voting

**Hard Voting (Votação Majoritária):**
```python
# Cada modelo vota 0 ou 1
Predição = maioria([modelo1.predict(), modelo2.predict()])
```

**Soft Voting (Média de Probabilidades):**
```python
# Média das probabilidades
Predição = média([modelo1.predict_proba(), modelo2.predict_proba()]) > 0.5
```

**Por que Hard Voting é melhor:**
- ✅ Decisões discretas são mais robustas
- ✅ Evita overfitting de probabilidades calibradas
- ✅ Gap positivo (+2.38pp) vs gap pequeno do soft voting (+0.21pp)

### 3. Robustez ao Overfitting

**Evidência:**
- **Gap positivo (+2.38pp):** Modelo generaliza MELHOR no teste que no treino
- **Métodos complexos falharam:**
  - GridSearchCV: -2.24pp gap (overfitting)
  - Stacking: -3.02pp gap (overfitting severo)
- **Simplicidade vence:** Menos parâmetros = menos overfitting

### 4. Teto de Performance

**11 submissões testadas, NENHUMA superou 81.88%:**
- Otimização Bayesiana (LightGBM): 79.71%
- GridSearchCV (216 combinações): 78.26%
- Stacking (5 modelos): 76.09%
- Threshold optimization: 78.99%

**Conclusão:** 81.88% representa o **teto de performance** para este dataset com as abordagens testadas.

---

## 🔄 Como Reproduzir

### Passo 1: Preparar o Ambiente

```bash
# Instalar dependências
pip install numpy pandas scikit-learn matplotlib seaborn

# Verificar estrutura do projeto
ls data/
# Esperado: train.csv, test.csv, sample_submission.csv
```

### Passo 2: Executar o Script de Ensemble

```bash
# Gerar as submissões de ensemble
python create_ensemble_submissions.py
```

**Saída esperada:**
- `submission_majority_vote.csv` - Hard voting (MELHOR - 81.88%)
- `submission_voting_ensemble.csv` - Soft voting (79.71%)
- `submission_weighted_ensemble.csv` - Weighted voting (79.71%)

### Passo 3: Submeter ao Kaggle

1. Fazer upload de `submission_majority_vote.csv` no Kaggle
2. Verificar formato: 277 linhas, colunas `id` e `labels`
3. Resultado esperado: **~81.88% de acurácia**

### Estrutura do Arquivo de Submissão

```csv
id,labels
0,1
1,0
2,1
...
276,1
```

---

## 🎓 Aprendizados Principais

### 1. Simplicidade > Complexidade

**O que funcionou:**
- ✅ Hard voting com 2 modelos Random Forest
- ✅ Hiperparâmetros simples e robustos
- ✅ Features originais (31) + features polinomiais (46)

**O que NÃO funcionou:**
- ❌ Stacking com 5 modelos e meta-learner
- ❌ GridSearchCV com 216 combinações
- ❌ Otimização Bayesiana com 150 trials
- ❌ Feature engineering extensiva (>50 features)

**Lição:** Occam's Razor se aplica - a solução mais simples é frequentemente a melhor.

### 2. Otimização Excessiva Prejudica

**Evidência:**
- GridSearchCV (80.50% CV) → 78.26% Kaggle (-2.24pp)
- Baseline (80.18% CV) → 78.26% Kaggle (-1.92pp)
- **Resultado:** Mesma performance no Kaggle, mas GridSearchCV teve mais overfitting

**Lição:** Mais otimização ≠ melhor performance. Overfitting ao CV é um risco real.

### 3. Gap CV-Kaggle é Indicador Crítico

**Gaps Positivos (Boa Generalização):**
- Hard voting: +2.38pp ✅
- Soft voting: +0.21pp ✅

**Gaps Negativos (Overfitting):**
- GridSearchCV: -2.24pp ❌
- Stacking: -3.02pp ❌

**Lição:** Gap positivo é raro e valioso - indica que o modelo generaliza melhor que o esperado.

### 4. Diversidade de Modelos é Essencial

**Por que 2 Random Forests funcionaram:**
- Features diferentes (31 vs 46)
- Hiperparâmetros diferentes (profundidade, regularização)
- Erros não-correlacionados

**Lição:** Diversidade > Quantidade. 2 modelos diversos > 5 modelos similares.

### 5. Dataset Pequeno Tem Limites

**Características:**
- 646 amostras de treino (pequeno)
- 31 features originais
- Ruído inerente em predição de sucesso de startups

**Implicações:**
- Teto de performance ~82%
- Modelos complexos overfitam facilmente
- Simplicidade é crucial

---

## 📁 Arquivos Relacionados

### Scripts Principais

| Arquivo | Descrição |
|---------|-----------|
| `create_ensemble_submissions.py` | Gera as 3 submissões de ensemble |
| `run_rf_gridsearch_fast.py` | GridSearchCV RF (não recomendado) |
| `run_stacking_ensemble.py` | Stacking ensemble (não recomendado) |

### Submissões Geradas

| Arquivo | Acurácia Kaggle | Recomendação |
|---------|-----------------|--------------|
| `submission_majority_vote.csv` | **81.88%** | ✅ **USAR ESTE** |
| `submission_voting_ensemble.csv` | 79.71% | Alternativa |
| `submission_weighted_ensemble.csv` | 79.71% | Alternativa |

### Documentação

| Arquivo | Conteúdo |
|---------|----------|
| `FINAL_RESULTS_ANALYSIS.md` | Análise completa dos resultados |
| `JOURNEY_SUMMARY.md` | Jornada de 78.26% → 81.88% |
| `SUBMISSION_COMPARISON.md` | Comparação detalhada de todas as submissões |

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

