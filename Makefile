# Makefile for unittest and linter.

.PHONY: test
test: code style lint

.PHONY: code
code:
	pytest

.PHONY: coverage
coverage:
	rm -rf coverage_html_report
	coverage run -m unittest discover -s tests
	coverage html

.PHONY: lint
lint:
	pylint turbofan
	bandit -r turbofan -ll

.PHONY: style
style:
	flake8
	mypy turbofan
	pycodestyle turbofan
	pydocstyle turbofan
