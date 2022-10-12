import pickle
import pandas as pd


# Paths.
PATH_TO_SAVE_NODE_FEATURE_DATA_TO = '../../datasets/gdsc/my_datasets/'
PATH_TO_SAVE_STRING_DATA_TO = '../../datasets/string/my_datasets/'

# File names.
GENE_EXPR_FINAL_FILE_NAME = 'joined_gdsc_geneexpr.pkl'
CNV_GISTIC_FINAL_FILE_NAME = 'joined_gdsc_cnv_gistic.pkl'
CNV_PICNIC_FINAL_FILE_NAME = 'joined_gdsc_cnv_picnic.pkl'
DRUG_FPS_FINAL_FILE_NAME = 'drug_name_fingerprints_dataframe.pkl'

# Specific paths.
PATH_TO_SAVED_DRUG_FEATURES = '../../datasets/gdsc/my_datasets/'
PATH_TO_SAVED_CL_FEATURES = '../../datasets/gdsc/my_datasets/'

# SPARSED FEATURE SETS
PATH_TO_FEATURES = '../../datasets/datasets_for_model_building/'

# --------------- #
# GENERAL METHODS #
# --------------- #
def load_pickle(file_name):
    return pd.read_pickle(file_name)
