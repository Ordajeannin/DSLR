# ========================
# CONFIG
# ========================

PYTHON = python3

DATASET = datasets/dataset_train.csv

DESCRIBE = data_analysis/describe.py
HISTOGRAM = data_visualization/histogram.py
SCATTER = data_visualization/scatter_plot.py
PAIR = data_visualization/pair_plot.py


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

all: describe histogram scatter pair

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

re: clean all

.PHONY: describe histogram scatter pair all clean re