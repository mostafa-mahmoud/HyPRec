#!/usr/bin/env sh

pylint --generate-rcfile > .lintrc
checked=false
for py_file in `find . -name "*.py"`
do
  if $checked
    then echo "\n"
  fi
  echo Linting\ $py_file
  lint_out=`pylint $py_file --reports=n --rcfile=.lintrc`
  if [ "$lint_out" = "" ]
  then
    lint_out="$py_file no errors."
  fi
  echo "$lint_out"
  checked=true;
done
rm .lintrc
echo "\nLinting done...\n"
