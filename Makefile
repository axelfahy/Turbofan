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
	pylint src
	bandit -r src -ll

.PHONY: style
style:
	flake8
	mypy src
	pycodestyle src
	pydocstyle src
