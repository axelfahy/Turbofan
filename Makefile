.PHONY: all clean test

TURBOFAN_URL = "https://ti.arc.nasa.gov/c/6/"

.PHONY: data/raw
data/raw:
	dvc run -n download \
		    -d src/data/download.py \
	        -d $(TURBOFAN_URL) \
			-o $@ \
		    python src/data/download.py $(TURBOFAN_URL) $@

.PHONY: data/interim
data/interim:
	dvc run -n create_dataset \
		    -d src/data/create_dataset.py \
			-d data/raw \
			-o $@ \
			python src/data/create_dataset.py data/raw $@

.PHONY: data/processed
data/processed:
	dvc run -n preprocess \
		    -d src/data/preprocess.py \
			-d data/interim \
			-o $@ \
			python src/data/preprocess.py data/interim $@

.PHONY: lint
lint:
	pylint src
	bandit -r src -ll

.PHONY: style
style:
	flake8
	mypy src
	pycodestyle src
	pydocstyle src
