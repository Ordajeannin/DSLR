# ========================
# CONFIG
# ========================

PYTHON = python3

DATASET = datasets/dataset_train.csv
DATATEST = datasets/dataset_test.csv

DESCRIBE = data_analysis/describe.py
HISTOGRAM = data_visualization/histogram.py
SCATTER = data_visualization/scatter_plot.py
PAIR = data_visualization/pair_plot.py

TRAIN_SGD = logistic_regression_SGD/logreg_train.py 
PREDICT_SGD = logistic_regression_SGD/logreg_predict.py
MODEL_SGD = modelSGD.json

TRAIN_BATCH = logistic_regression/logreg_train.py
PREDICT = logistic_regression/logreg_predict.py
MODEL_BATCH = model.json


# ========================
# RULES
# ========================

describe:
	$(PYTHON) $(DESCRIBE) $(DATASET)

histogram:
	$(PYTHON) $(HISTOGRAM) $(DATASET)

scatter:
	$(PYTHON) $(SCATTER) $(DATASET)

pair:
	$(PYTHON) $(PAIR) $(DATASET)

train:
	$(PYTHON) $(TRAIN_BATCH) $(DATASET)

predict: train
	$(PYTHON) $(PREDICT) $(DATATEST) $(MODEL_BATCH) 

train_sgd:
	$(PYTHON) $(TRAIN_SGD) $(DATASET)

predict_sgd: train_sgd
	$(PYTHON) $(PREDICT_SGD) $(DATATEST) $(MODEL_SGD)

all: describe histogram scatter pair

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

re: clean all

.PHONY: describe histogram scatter pair all clean re