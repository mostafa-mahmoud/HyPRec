#!/usr/bin/env python3
"""
A script that runs the tests of the project.
"""
import sys
import os
import unittest


def runtests(test_args=None):
    """
    Initialize a test suite that contains all the tests in the given modules
    in @test_args, and run the test suite. The modules are taken by command
    line arguments.
    """
    if test_args is not None:
        base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tests')
        test_args = ['tests.' + os.path.splitext(f)[0] for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    suite = unittest.TestSuite()
    for test in test_args:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(test))
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    runtests(sys.argv[1:])
