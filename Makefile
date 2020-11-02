.PHONY: all clean test

TURBOFAN_URL = "https://ti.arc.nasa.gov/c/6/"

.PHONY: data/raw
data/raw:
	python src/data/download.py $(TURBOFAN_URL) $@