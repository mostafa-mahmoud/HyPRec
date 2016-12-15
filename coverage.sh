#!/usr/bin/env sh

python-coverage run runtests.py
python-coverage report > coverage_report.txt
cat coverage_report.txt
