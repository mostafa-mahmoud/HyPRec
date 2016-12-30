#!/usr/bin/env python3
"""
A script that runs the tests of the project.
"""
import sys
import unittest


def runtests(test_args=['tests.tests']):
    """
    Initialize a test suite that contains all the tests in the given modules
    in @test_args, and run the test suite. The modules are taken by command
    line arguments.
    """
    if not test_args:
        test_args = ['tests.tests']
    suite = unittest.TestSuite()
    for test in test_args:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(test))
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    runtests(sys.argv[1:])
