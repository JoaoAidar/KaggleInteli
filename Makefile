.PHONY: eda cv tune train train-best submit all clean tune-advanced tpot ensemble advanced-all

DATA_DIR=data
SEED=42
MODEL=rf
N_ITER=100
GENERATIONS=5
POPULATION_SIZE=20

# Basic pipeline
eda:
	python -m src.cli eda --data-dir $(DATA_DIR)

cv:
	python -m src.cli cv --data-dir $(DATA_DIR)

tune:
	python -m src.cli tune --data-dir $(DATA_DIR) --seed $(SEED)

train:
	python -m src.cli train-predict --data-dir $(DATA_DIR) --model $(MODEL) --seed $(SEED)

train-best:
	python -m src.cli train-predict --data-dir $(DATA_DIR) --use-best-rf --seed $(SEED)

submit: train-best
	@echo "✓ submission.csv generated and ready for Kaggle upload"

all: eda cv tune submit

# Advanced pipeline
tune-advanced:
	python -m src.cli tune-advanced --data-dir $(DATA_DIR) --seed $(SEED) --n-iter $(N_ITER) --models rf xgb lgb

tune-advanced-fe:
	python -m src.cli tune-advanced --data-dir $(DATA_DIR) --seed $(SEED) --n-iter $(N_ITER) --models rf xgb lgb --engineer-features

tpot:
	python -m src.cli tpot --data-dir $(DATA_DIR) --seed $(SEED) --generations $(GENERATIONS) --population-size $(POPULATION_SIZE)

tpot-fe:
	python -m src.cli tpot --data-dir $(DATA_DIR) --seed $(SEED) --generations $(GENERATIONS) --population-size $(POPULATION_SIZE) --engineer-features

ensemble:
	python -m src.cli ensemble --data-dir $(DATA_DIR) --seed $(SEED)

ensemble-fe:
	python -m src.cli ensemble --data-dir $(DATA_DIR) --seed $(SEED) --engineer-features

# Complete advanced pipeline
advanced-all: eda cv tune-advanced-fe ensemble-fe
	@echo "✓ Advanced pipeline complete!"

# Cleaning
clean:
	rm -f submission.csv
	rm -f reports/cv_metrics.csv
	rm -f reports/best_rf_params.json
	rm -f reports/best_rf_extensive_params.json
	rm -f reports/best_xgb_params.json
	rm -f reports/best_lgb_params.json
	rm -f reports/advanced_tuning_summary.json
	rm -f reports/ensemble_metrics.csv
	rm -f reports/tpot_pipeline.py

