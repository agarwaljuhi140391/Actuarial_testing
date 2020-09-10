##model script to provide outputs
import os
import sys
import pandas as pd
import numpy as np
from src.dais.models.traditional2 import TraditionalLifeFunctions
from src.dais.models.common3 import BaseModel, BondModel
import abc
from src import dais
sys.path.insert(0, os.path.join(os.getcwd() + '/src'))

class execute_graph(abc.ABC):
    def __init__(self,in_folder):
        self.a = BaseModel()
        self.a.setfuncs(TraditionalLifeFunctions)
        self.mpf=pd.read_table(os.path.join(in_folder,'MPF.txt'),header=0)
        def cp_type(x):
            if x[:1]=="A":
                return 0
            elif x[:2]=="L-":
                return 1
            elif x[:2]=="Le":
                return 2
            else:
                return 999999
        def cp_type_adj(x):
            if x[:1]=="A":
                return 0
            elif x[:2]=="L-":
                return 1
            elif x[:2]=="Le":
                return 2
            elif x[:2]=="J-":
                return 3
            else:
                return 999999
        self.mpf=self.mpf.assign(BASE_PROD_CODE_patch=self.mpf['BASE_PROD_CODE'].str.split('_').str.get(-1).astype(np.int64)
            ,CP_TYPE_ADJ_patch=self.mpf['CP_TYPE_ADJ'].apply(cp_type_adj)
            , CP_TYPE_patch=self.mpf['CP_TYPE'].apply(cp_type)
            , ISS_YR_IDX=999999)
        self.a.add_source_mpf(self.mpf)
        self.a.add_source_tables(pd.read_excel(os.path.join(in_folder,'TablesBE.xlsx'),sheet_name=None))
        self.a.add_source_tables(pd.read_excel(os.path.join(in_folder,'TablesCommon.xlsx'),sheet_name=None))
        CONSTANTS={'max_incidence_lkup_age': 126,
                   'max_mort_sel_lkup_yr': 25,
                   'max_lapse_lkup_yr': 21,
                   'min_ppia_pc_lkup_yr':1998,
                   'max_ppia_pc_lkup_yr':2017,
                   'val_yr':2017,
                   'val_mth':12,
                   'BE_INDEX':2,
                   'COMM_LIMIT_ADJ_index':1,
                   'IC_PERIOD_M':12,
                   'MIN_TAX_LKUP_Y':1996,
                   'MAX_TAX_LKUP_Y':2020,
                   'MAX_COMM_MILE_M':120,
                   'COMM_MILE_FAC':0.25,
                   'RC_START_M':13,
                   'MIN_COMM_ID':5,
                   'M_DISC_RATE':(1+0.06)**(1/12)-1,
                   't_range':1200,
                   'SPIKE_RATE_PERIOD_INDEX':1,
                   'MORT_PAD_PC':0,
                   'DEC_WOP_PAD_PC':0,
                   'MAX_EXP_LKUP_Y':20,
                   'ASSPT_BE_index':1,
                   'COM_YEAR_END':3,
                   'COMM_CB_ST_YM':201510,
                   'COMM_CB_ADJ_RATE':0.4,
                   'MAX_COMM_CB_M':48}
        self.a.add_source_constants(CONSTANTS)
        self.a.initialise()
        self.a.execute(dokeep='ALL')
        
    def get_outputs(self,list_node_names=None):
        if list_node_names is not None:
            out_dict = { key: self.a.vals[key] for key in list_node_names }
        else:
            out_dict = self.a.vals
        return out_dict        
  