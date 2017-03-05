Master project about recommender systems.

Software for development
========================
  * pep8
  * python-coverage
  * mysql-client
  * mysql-server
  * mysql-python
  * json
  * sklearn
  * numpy
  * sphinx
  * scipy
  * overrides
  * tensorflow
  * keras
  
  * pandas
  * lda2vec
  * spaCy
  * chainer
  * python-dev


Testing
=======
#. Running the runtests.py script, will run the tests in tests.tests ::

      python runtests.py

#. Running the runtests.py script, with arguments that contain the modules of the unittests, will run the tests in all the provided modules. ::

      python runtests.py test1 test2 test3 # ...

Database
========
Execute these 2 statements in MySQL console before importing the data ::

      set autocommit = 0;
