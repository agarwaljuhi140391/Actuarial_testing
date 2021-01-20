  
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
    def __init__(self,testname,inp,exp_out):
        super(OutputTest, self).__init__(testname)
    def test_POL_YR(self):
        #for key in self.td.keys():
        #    if np.array_equal(self.td.get(key).round(decimals=12),self.d.get(key).round(decimals=12)):
        #        self.match[key]=1
        #    else:
        #        self.match[key]=0
        #match_keys = [key for key in self.td.keys() if self.match[key]==1]
        #unmatched_keys = [key for key in self.td.keys() if self.match[key]==0]
        #print("Matched fields: ",match_keys)
        #print("Unmatched fields: ", unmatched_keys)
        out = TraditionalLifefunctions.POL_YR(inp)
        self.assertTrue(out==exp_out)


if __name__ == '__main__':
    #args = parser.parse_args()
    #df1 = pd.read_excel(args.ground_truth_filename, 'static', header=None)
    #td1={}
    #for i in range(len(df1)):
    #    td1[df1.iloc[i,0]] = np.array(df1.iloc[i,1]).reshape(1,1)
    #df2 = dict(pd.read_excel(args.ground_truth_filename, 'dynamic'))
    #td2 = {key:df2[key].values[:1200].reshape(1,1200) for key in df2.keys()}
    #td = {**td1, **td2}
    #g = execute_graph(args.test_data_folder)
    #d = g.get_outputs()
    #unittest.main()
    #suite = unittest.TestSuite()
    #suite = unittest.defaultTestLoader.loadTestsFromTestCase(OutputTest(testname))
    #suite.addTest(OutputTest(d, td))
    #unittest.TextTestRunner(verbosity=2).run(suite)
    #python unittest testcase1.py
    f = open('testcase.json',)
    d = json.load(f)
    
    test_loader = unittest.TestLoader()
    #test_names = test_loader.getTestCaseNames(OutputTest)
    suite = unittest.TestSuite()
    for id in len(d):
        func_name = d['testcase'][id]['func_name']
        inp = d['testcase'][id]['inputs']
        exp_out = d['testcase'][id]['outputs']
        suite.addTest(OutputTest(str('test_' + func_name), inp,exp_out))
    unittest.TextTestRunner().run(suite)
    
    
    
    
