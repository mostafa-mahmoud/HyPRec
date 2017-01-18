.PHONY: clean-pyc clean-build docs help
.DEFAULT_GOAL := all
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

all: clean lint test

clean: clean-pyc

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint: ## check style with pep8
	pep8 data lib tests util runnables.py

lint_flake: ## check style with flake8
	flake8 data lib tests util runnables.py

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/lib.rst
	rm -f docs/util.rst
	sphinx-apidoc -o docs/ lib
	sphinx-apidoc -o docs/ util
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

test: ## run tests quickly with the default Python
	python3 runtests.py

run: ## run recommender
	python3 runnables.py -lsvd

remove_training: ## removing training models
	find matrices -name '*.dat' -exec rm -f {} +

rebuild_database: ## rebuild the database
	python3 -c "from util.data_parser import DataParser; DataParser.drop_database(); DataParser.process()"

coverage: ## check code coverage quickly with the default Python
	python-coverage run runtests.py
	python-coverage report -m > coverage_report.txt
	cat coverage_report.txt
	python-coverage html
	open htmlcov/index.html