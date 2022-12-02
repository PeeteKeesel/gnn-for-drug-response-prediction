import os        
import shutil
import urllib.request
import pandas as pd
import numpy as np
 
from zipfile import ZipFile
from enum import Enum
from src.utils.preprocess_helper import get_gdsc_gene_expression


class DownloadLinks(Enum):
    GDSC1 = 'ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC1_fitted_dose_response_24Jul22.xlsx'
    GDSC2 = 'ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC2_fitted_dose_response_24Jul22.xlsx'
    GEXPR = 'https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip'
    CL_DETAILS = 'ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/Cell_Lines_Details.xlsx'  
    CNV = 'https://cog.sanger.ac.uk/cmp/download/cnv_20191101.zip'
    MUT = 'https://cog.sanger.ac.uk/cmp/download/mutations_all_20220315.zip'
    PROTEIN_LINKS = 'https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz'
    PROTEIN_INFO = 'https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz'
    
    @classmethod
    def get_names(cls):
        return [data.name for data in cls]


class Processor:
    def __init__(self,
                 raw_path: str, 
                 processed_path: str,
                 download_links=DownloadLinks):
        """Created class to download, process and create all raw files.
        """
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.download_links = download_links

        # File names of the saved raw datasets.
        # 1. Files which need to be downloaded.
        self.raw_gdsc1_file = 'GDSC1_fitted_dose_response_24Jul22.xlsx'
        self.raw_gdsc2_file = 'GDSC2_fitted_dose_response_24Jul22.xlsx'
        self.raw_landmark_file = 'landmark_genes.csv'
        self.raw_gexpr_file = 'Cell_line_RMA_proc_basalExp.txt'
        self.raw_cl_details_file = 'Cell_Lines_Details.xlsx'
        self.raw_cnvp_file = 'cnv_abs_copy_number_picnic_20191101.csv'
        self.raw_cnvg_file = 'cnv_gistic_20191101.csv'
        self.raw_mut_file = 'mutations_all_20220315.csv'
        self.raw_protein_links_file = '9606.protein.links.detailed.v11.5.txt.gz'
        self.raw_protein_info_file = '9606.protein.info.v11.5.txt.gz'
        
        # 2. Additional file names which don't need to be downloaded.
        self.raw_smiles_file = 'GDSC_compounds_inchi_key_with_smiles.csv'
        self.raw_landmark_genes_file = 'landmark_genes.csv'        
        
        self.landmark_genes = {}
        self.landmark_genes_df = None # pd.Series?

    # ---------- #
    # DOWNLOADER #
    # ---------- #
    def _download_from_link(self, name: str, value: str):
        """Download file(s) from the given list and return the saved file name."""
        file_name = value.split('/')[-1]
        print(f"Downloading from {value}")
        urllib.request.urlretrieve(value, self.raw_path + file_name)
        print(f"Finished download into {self.raw_path + file_name}.")
        return file_name
    
    def _extract_zip_if_necessary(self, file_name):
        """Extract the given file if it is a zip and return the extracted files
        if there are any."""
        extracted_files = []
        if file_name.endswith('.zip'):
            with ZipFile(self.raw_path + file_name, 'r') as zipf:
                print(f"{4*' '}Extrating zip file {self.raw_path + file_name} ...")
                zipf.extractall(self.raw_path)
                extracted_files = zipf.namelist()
                for ef in extracted_files:
                    print(f"{8*' '}Extracted file: {os.path.join(self.raw_path, ef)}")
        return extracted_files

    def download_raw_datasets(self):
        """Downloads provided dataset into the raw path."""
        for name in self.download_links.get_names():
            print(f"{20*'='}\nDownloading {name}...")
            file_name = self._download_from_link(name.lower(), self.download_links[name].value)
            
            # If the file is a zip, extract it.
            extracted_files = self._extract_zip_if_necessary(file_name)
            
            match name:
                case 'GDSC1': 
                    self.raw_gdsc1_file = file_name 
                case 'GDSC2': 
                    self.raw_gdsc2_file = file_name
                case 'GEXPR':
                    assert len(extracted_files) == 1, \
                        f"Gene expression zip {name} should contain only 1 " \
                        + f"file but countains {len(extracted_files)}."
                    self.raw_gexpr_file = extracted_files[0]
                case 'CL_DETAILS': 
                    self.raw_cl_details_file = file_name
                case 'CNV': 
                    assert len(extracted_files) == 2, \
                        f"Copy number variation zip {name} should contain 2 " \
                        + f"files but contains {len(extracted_files)}."
                    self.raw_cnvg_file = [f for f in extracted_files if 'gistic' in f]
                    self.raw_cnvp_file = [f for f in extracted_files if 'picnic' in f]                    
                case 'MUT': 
                    assert len(extracted_files) == 1, \
                        f"Mutations zip {name} should contain only 1 " \
                        + f"file but countains {len(extracted_files)}."
                    self.raw_mut_file = extracted_files[0]
                case 'PROTEIN_LINKS': 
                    self.raw_protein_links_file = file_name
                case 'PROTEIN_INFO': 
                    self.raw_protein_info_file = file_name
                # case 'LANDMARK_GENES': self.raw_landmark_file = file_name # TODO: are these coming from a download?

    def _add_additional_datasets(self):
        """Copies the additional datasets to the raw dataset folder which 
        now holds all the raw datasets. This includes the downloaded raw
        datasets, as well as the additional datasets."""
        raw_path_additional = 'data/additional/' 
        
        print(f"{20*'='}\nCopying already existing {self.raw_smiles_file} from {raw_path_additional} ...")
        shutil.copyfile(raw_path_additional + self.raw_smiles_file, 
                        self.raw_path + self.raw_smiles_file)
        print(f"Finished copying into {self.raw_path + self.raw_smiles_file}")
        
        print(f"{20*'='}\nCopying already existing {self.raw_landmark_genes_file} from {raw_path_additional} ...")
        shutil.copyfile(raw_path_additional + self.raw_landmark_genes_file, 
                        self.raw_path + self.raw_landmark_genes_file)
        print(f"Finished copying into {self.raw_path + self.raw_landmark_genes_file}")
                
    def create_raw_datasets(self):
        self.download_raw_datasets()
        self._add_additional_datasets()

    # ------------ #
    # PREPROCESSOR #
    # ------------ #    
    def _process_gdsc_fitted(self):
        gdsc1 = pd.read_excel(self.raw_path + self.raw_gdsc1_file, header=0)
        gdsc2 = pd.read_excel(self.raw_path + self.raw_gdsc2_file, header=0)

        gdsc_join = pd.concat([gdsc1, gdsc2], ignore_index=True)
        assert gdsc_join[gdsc_join.index.duplicated()].shape[0] == 0,\
            "There are duplicated rows in the joined GDSC dataset."
        assert gdsc_join.shape[0] == gdsc1.shape[0] + gdsc2.shape[0],\
            "The joined GDSC dataset is not the concatenation of the two GDSC sets."

        cols_to_keep = ['DATASET', 'CELL_LINE_NAME', 'DRUG_NAME', 'DRUG_ID', 
                        'SANGER_MODEL_ID', 'AUC', 'RMSE', 'Z_SCORE', 'LN_IC50']
        gdsc_base = gdsc_join[cols_to_keep].drop_duplicates()
        gdsc_base.to_pickle(self.process_path + 'drm_full.pkl')
        print(f"Successfully saved full GDSC dataset in `{self.process_path + 'drm_full.pkl'}`.")

        del gdsc1, gdsc2, gdsc_join, cols_to_keep, gdsc_base

    
    def _set_landmark_genes(self):
        landmark_genes = pd.read_csv(self.raw_path + self.raw_landmark_file, sep="\t")   
        self.landmark_genes = set(landmark_genes.Symbol.values.tolist())
        self.landmark_genes_df = landmark_genes.Symbol

        del landmark_genes

    
    def _process_gene_expression(self):
        gexpr = get_gdsc_gene_expression(path_cell_annotations=self.raw_path + self.raw_cl_details_file,
                                         path_gene_expression=self.raw_path + self.raw_gexpr_file)
           
        # Choose only the cell-line columns of the gene expressions table that are in the landmark gene file.
        gexpr_sparse = gexpr[list(set(gexpr.columns).intersection(self.landmark_genes))]
        gexpr_sparse.columns.rename('CELL_LINE_NAME', inplace=True)

        # Read already processed drug response matrix.
        drm = pd.read_pickle(self.process_path + 'drm_full.pkl')

        # Get gene expression level per cell in the drug-response matrix.
        gdsc_full = drm.merge(right=gexpr_sparse,
                              left_on=['CELL_LINE_NAME'],
                              right_index=True,
                              how='left',
                              suffixes=['_gdsc', '_geneexpr'])
        gdsc_full.to_pickle(self.process_path + 'gexpr_full.pkl')
        print(f"Successfully saved full Gene Expression dataset in `{self.process_path + 'gexpr_full.pkl'}`.")        

        del gexpr, gexpr_sparse, drm, gdsc_full

    
    def _process_copy_number_variation(self, cnv_type: str):
        # Process copy number picnic.
        cnv = None
        if cnv_type == 'cnvp':
            cnv = pd.read_csv(self.raw_path + self.raw_cnvp_file, sep=",", header=1)
        elif cnv_type == 'cnvg':
            cnv = pd.read_csv(self.raw_path + self.raw_cnvg_file, sep=",", header=1)            
        else:
            raise ValueError(f"ERROR: Given cnv_type {cnv_type} needs to be in ['cnvp', 'cnvg'].")

        cnv.rename(columns={
            'Unnamed: 1': 'GENE_SYMBOL',
            'model_name': 'GENE_ID'}, inplace=True)
        cnv = cnv.iloc[1:, :]

        # Get the genes are column names.
        cnv2 = cnv.iloc[:, 1:].T
        cnv2 = cnv2.rename(columns=cnv2.iloc[0]).drop(cnv2.index[0])
        del cnv

        # Select only the cell-line columns of the CNV dataset which are landmark genes.
        cols_to_keep = [c for c in cnv2.columns[cnv2.columns != 'nan'] if c in self.landmark_genes]
        cnv3 = cnv2[cols_to_keep]
        assert cnv3.shape[1] == len(cols_to_keep),\
            "ERROR: The CNV got not correctly selected."
        del cnv2, cols_to_keep

        # Read already processed drug response matrix.
        drm = pd.read_pickle(self.process_path + 'drm_full.pkl')

        # Get copy number variation per cell in the drug-response matrix.
        cnv_full = drm.merge(right=cnv3,
                             left_on=['CELL_LINE_NAME'],
                             right_index=True,
                             how='left',
                             suffixes=['_gdsc', f'_{cnv_type}'])

        cnv_full.to_pickle(self.process_path + f'{cnv_type}_full.pkl')
        print(f"Successfully saved full CNV {cnv_type} dataset in `{self.process_path}{cnv_type}_full.pkl`.")

        del cnv3, drm, cnv_full

    
    def _process_mutations(self):
        mut = pd.read_csv(self.raw_path + self.raw_mut_file, sep=",", header=0)

        # Read already processed drug response matrix.
        drm = pd.read_pickle(self.process_path + 'drm_full.pkl') 

        # Find the CELL_LINE_NAME's per SANGER_MODEL_ID.
        celllines_per_sangermodelid = drm[['SANGER_MODEL_ID', 'CELL_LINE_NAME']]\
            .groupby('SANGER_MODEL_ID')['CELL_LINE_NAME'].nunique()
        counts_per_sangermodelid = celllines_per_sangermodelid.values
        assert (counts_per_sangermodelid == 1).all(),\
            "ERROR: Not all Sanger Model ID's have only one count."

        # Only take the interested columns for the mapping.
        gdsc_mapping_subset = drm[['SANGER_MODEL_ID', 'CELL_LINE_NAME']]

        # Only take the unique SANGER_MODEL_ID's, since these have a 1-to-1 relationship to the CELL_LINE_NAME's anyways.
        gdsc_mapping_subset = gdsc_mapping_subset.groupby('SANGER_MODEL_ID').first().reset_index(level=0)

        # Join the CELL_LINE_NAME's onto the mutations_all dataset, based on the model_id.
        mut2 = mut.merge(right=gdsc_mapping_subset,
                         left_on='model_id',
                         right_on='SANGER_MODEL_ID',
                         how='left')
        mut3 = mut2[mut2['CELL_LINE_NAME'].notna()]
        del mut, mut2

        # Take only the rows which have a `gene_symbol` which is also present in the landmark genes table.
        mut4 = mut3.merge(right=self.landmark_genes_df,
                          left_on='gene_symbol',
                          right_on='Symbol')        
        del mut3

        mut5 = mut4[['CELL_LINE_NAME',
                     'gene_symbol',
                     'model_id',
                     'protein_mutation',
                     'rna_mutation',
                     'cdna_mutation',
                     'cancer_driver',
                     'vaf']]
        del mut4

        mut6 = pd.pivot_table(data=mut5,
                              values='cancer_driver',
                              index=['CELL_LINE_NAME'],
                              columns=['gene_symbol'],
                              aggfunc=np.sum,
                              dropna=False)
        del mut5

        # Set mutation values: 1.0=mutation, 0.0=no_mutation
        mut6[mut6 == 0.0] = 1.0
        mut6[np.isnan(mut6)] = 0.0

        mut6['CELL_LINE_NAME'] = mut6.index
        mut6.insert(0, 'CELL_LINE_NAME', mut6.pop('CELL_LINE_NAME'))
        mut6.reset_index(drop=True, inplace=True)
        mut6.columns.name = 'GENE_SYMBOL'        

        # Get mutational information per cell in the drug-response matrix.
        mut_full = drm.merge(right=mut6,
                             on=['CELL_LINE_NAME'],
                             how='inner',
                             suffixes=['_gdsc', '_mut'])
        
        mut_full.to_pickle(self.process_path + 'mut_full.pkl')
        print(f"Successfully saved full Mutations dataset in `{self.process_path + 'mut_full.pkl'}`.")

        del mut6, drm, mut_full         

    
    def process_raw(self):
        self._process_gdsc_fitted()
        self._set_landmark_genes()
        self._process_gene_expression()
        self._process_copy_number_variation('cnvp')
        self._process_copy_number_variation('cnvg')
        self._process_mutations()

    
    def _process_proteins(self):
        return NotImplementedError