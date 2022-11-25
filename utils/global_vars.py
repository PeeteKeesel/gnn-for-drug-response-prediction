from enum import Enum

PATH_TO_GDSC_SCREENING_DATA = '../../datasets/gdsc/screening_data/'

# Paths.
PATH_TO_SAVE_NODE_FEATURE_DATA_TO = '../../datasets/gdsc/my_datasets/'
PATH_TO_SAVE_STRING_DATA_TO = '../../datasets/string/my_datasets/'

# File names.
GENE_EXPR_FINAL_FILE_NAME = 'joined_gdsc_geneexpr.pkl'
CNV_GISTIC_FINAL_FILE_NAME = 'joined_gdsc_cnv_gistic.pkl'
CNV_PICNIC_FINAL_FILE_NAME = 'joined_gdsc_cnv_picnic.pkl'

# ------------------- #
# Plotting parameters #
# ------------------- #
class PlottingParameters(Enum):
    TITLE_FONTSIZE = 15
    XLABEL_FONTSIZE = 15
    YLABEL_FONTSIZE = 15
    XTICKS_LABELSIZE = 13
    YTICKS_LABELSIZE = 13