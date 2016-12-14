#!/usr/bin/env python
import unittest


class TestcaseBase(unittest.TestCase):
    def setUp(self):
        """
        Setup method that is called at the beginning of each test.
        """
        pass


class SampleTest(TestcaseBase):
    def runTest(self):
        """
        The test method.
        """
        self.assertEqual(2 + 2, 4, 'Obvious truth is wrong')
