This unitest test is for testing some static and dynamic fields calculated using the src code.
In this example we use 2 static fields and 6 dynamic fields for testing.
dict_keys(['PREM_FREQ', 'MORT_SMK_IDX', 'IC_PAYBL_PP', 'RC_PAYBL_PP', 'PREM_INC', 'FISCAL_YEAR', 'RC_PAID', 'ANN_PREM_PP'])

we need to provide input data for source code to compute these values. Input data should be entered in MPF.txt file in test_data_folder for a single policy.
Also we need ground truth (actual values) of these fields to comare against the calculated ones. We get these values from the excel macros and place these in ground_truth.xlsx file.

We run the test and provide output as follows:
======================================================================
FAIL: test_run (__main__.OutputTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "testcase1.py", line 26, in test_run
    self.assertTrue(len(match_keys)==len(self.td.keys()))
AssertionError: False is not true

----------------------------------------------------------------------
Ran 1 test in 0.000s


###Next Steps is to add more tests in this test suite like:
1. Check whether the input data maps all the graph nodes
2. Test related to data type of input fields
3. Test for graph properties (NODE COLOUR, dependencies etc.)
4. Test individual class methods 
