#!/usr/bin/env sh

for py_file in `find . -name "*.py"`
do
  pep8 $py_file
done
echo "\nLinting done...\n"
