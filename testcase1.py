import unittest
import argparse
import os
import sys
import pandas as pd
import numpy as np
from model import execute_graph

parser = argparse.ArgumentParser(description="argument parser")
parser.add_argument("--test_data_folder", type = str, default = str(os.getcwd()) + '/test_data_folder', required=False, help="specify folder path of test data")
parser.add_argument("--ground_truth_filename", type = str, default = 'ground_truth.xlsx', required=False, help="specify xlsx filename that contains groundtruth")

class OutputTest(unittest.TestCase):
    def __init__(self,testname,d,td):
        super(OutputTest, self).__init__(testname)
        self.d=d
        self.td=td
        self.match = {}
    def test_run(self):
        for key in self.td.keys():
            if np.array_equal(self.td.get(key),self.d.get(key).squeeze()):
                self.match[key]=1
            else:
                self.match[key]=0
        match_keys = [key for key in self.td.keys() if self.match[key]==1]
        self.assertTrue(len(match_keys)==len(self.td.keys()))


if __name__ == '__main__':
    args = parser.parse_args()
    df1 = pd.read_excel(args.ground_truth_filename, 'static', header=None)
    td1={}
    for i in range(len(df1)):
        td1[df1.iloc[i,0]] = df1.iloc[i,1]
    df2 = dict(pd.read_excel(args.ground_truth_filename, 'dynamic'))
    td2 = {key:df2[key].values[:1200].reshape(1,1200) for key in df2.keys()}
    td = {**td1, **td2}
    g = execute_graph(args.test_data_folder)
    d = g.get_outputs()
    #unittest.main()
    #suite = unittest.TestSuite()
    #suite = unittest.defaultTestLoader.loadTestsFromTestCase(OutputTest(testname))
    #suite.addTest(OutputTest(d, td))
    #unittest.TextTestRunner(verbosity=2).run(suite)
    #python unittest testcase1.py
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(OutputTest)
    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(OutputTest(test_name, d,td))
    unittest.TextTestRunner().run(suite)

