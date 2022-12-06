import os        
import shutil
import torch
import pickle
import urllib.request
import pandas as pd
import numpy as np
 
from zipfile import ZipFile
from enum import Enum
from src.utils.preprocess_helper import get_gdsc_gene_expression

# For gene-gene interaction graph creation.
from torch_geometric.data import Data
from io import BytesIO
from typing import List
import gzip
from typing import Tuple, Set, FrozenSet
from tqdm import tqdm

# For drug graphs
from torch_geometric.utils import from_smiles 


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
                 download_links=DownloadLinks,
                 combined_score_thresh: int=700,
                 gdsc: str='gdsc2'):
        """Created class to download, process and create all raw files.
        Notes: 
            - all datasets which are independent of the GDSC database and the threshold 
                are saved in the `processed_path` folder.
            - all datasets which are dependent on the GDSC database but not on the
                threshold are saved in the `processed_path/gdsc/` folder. This includes
                the 
                - SMILES dataset and the 
                - Drug-Response Matrix.
            - all datasets which are dependent on the GDSC database and on the
                threshold are saved in the `processed_path/gdsc/combined_score_thresh/` 
                folder. This includes the 
                - Feature Datasets
        """
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.gdsc_path = self.processed_path + gdsc.lower() + '/'
        self.gdsc_thresh_path = self.gdsc_path + str(combined_score_thresh) + '/'
        
        self.download_links = download_links

        # File names of the saved raw datasets.
        # 1. Files which need to be downloaded.
        self.raw_gdsc1_file = 'GDSC1_fitted_dose_response_24Jul22.xlsx'
        self.raw_gdsc2_file = 'GDSC2_fitted_dose_response_24Jul22.xlsx'
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
        
        self.combined_score_thresh = combined_score_thresh
        self.gdsc = gdsc.lower()

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

    # ------------- #
    # PRE-PROCESSOR #
    # ------------- #
    def _process_gdsc_fitted(self):
        print(40*'=')
        print(f"{f'Processing GDSC datasets...':<40}")         
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
        gdsc_base.to_pickle(self.processed_path + 'drm_full.pkl')
        print(f"Successfully saved full GDSC datasets in {self.processed_path + 'drm_full.pkl'}")

        del gdsc1, gdsc2, gdsc_join, cols_to_keep, gdsc_base

    
    def _set_landmark_genes(self):
        landmark_genes = pd.read_csv(self.raw_path + self.raw_landmark_genes_file, sep="\t")   
        self.landmark_genes = set(landmark_genes.Symbol.values.tolist())
        self.landmark_genes_df = landmark_genes.Symbol

        del landmark_genes

    
    def _process_gene_expression(self):
        print(40*'=')
        print(f"{f'Processing gene expression dataset...':<40}")        
        gexpr = get_gdsc_gene_expression(path_cell_annotations=self.raw_path + self.raw_cl_details_file,
                                         path_gene_expression=self.raw_path + self.raw_gexpr_file)
           
        # Choose only the cell-line columns of the gene expressions table that are in the landmark gene file.
        gexpr_sparse = gexpr[list(set(gexpr.columns).intersection(self.landmark_genes))]
        gexpr_sparse.columns.rename('CELL_LINE_NAME', inplace=True)

        # Read already processed drug response matrix.
        drm = pd.read_pickle(self.processed_path + 'drm_full.pkl')

        # Get gene expression level per cell in the drug-response matrix.
        gdsc_full = drm.merge(right=gexpr_sparse,
                              left_on=['CELL_LINE_NAME'],
                              right_index=True,
                              how='left',
                              suffixes=['_gdsc', '_geneexpr'])
        
        print(f"Shape of gdsc_full: {self.gdsc_full.shape}")        
        gdsc_full.to_pickle(self.processed_path + 'gexpr_full.pkl')
        print(f"Successfully saved full Gene Expression dataset in {self.processed_path + 'gexpr_full.pkl'}.")

        del gexpr, gexpr_sparse, drm, gdsc_full

    
    def _process_copy_number_variation(self, cnv_type: str):
        which_cnv = 'gistic' if cnv_type[-1]=='g' else 'picnic'
        print(40*'=')
        print(f"{f'Processing copy number variation {which_cnv} dataset... ':<40}")
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
        drm = pd.read_pickle(self.processed_path + 'drm_full.pkl')

        # Get copy number variation per cell in the drug-response matrix.
        cnv_full = drm.merge(right=cnv3,
                             left_on=['CELL_LINE_NAME'],
                             right_index=True,
                             how='left',
                             suffixes=['_gdsc', f'_{cnv_type}'])

        print(f"Shape of {cnv_type}_full: {cnv_full.shape}")        
        cnv_full.to_pickle(self.processed_path + f'{cnv_type}_full.pkl')
        print(f"Successfully saved full CNV {which_cnv} dataset in {self.processed_path}{cnv_type}_full.pkl.")

        del cnv3, drm, cnv_full, which_cnv

    
    def _process_mutations(self):
        print(40*'=')
        print(f"{f'Processing mutations dataset...':<40}")
        mut = pd.read_csv(self.raw_path + self.raw_mut_file, sep=",", header=0)

        # Read already processed drug response matrix.
        drm = pd.read_pickle(self.processed_path + 'drm_full.pkl') 

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
        
        print(f"Shape of mut_full: {mut_full.shape}")
        mut_full.to_pickle(self.processed_path + 'mut_full.pkl')
        print(f"Successfully saved full Mutations dataset in `{self.processed_path + 'mut_full.pkl'}`.")

        del mut6, drm, mut_full         

    
    def _get_intersecting_subset(self,
                                 df: pd.DataFrame, 
                                 title: str, 
                                 inter_genes, 
                                 inter_cls):
        print(f"{title}\n{len(title)*'='}")
        df_inter = df[['DRUG_ID', 'DATASET', 'CELL_LINE_NAME'] + list(inter_genes)]
        df_inter = df_inter[df_inter.CELL_LINE_NAME.isin(list(inter_cls))]
        print(f"Shape total: {df_inter.shape}")
        print(f"Shape GDSC1: {df_inter[df_inter.DATASET=='GDSC1'].shape}")
        print(f"Shape GDSC2: {df_inter[df_inter.DATASET=='GDSC2'].shape}")
        return df_inter  
    
    def _get_only_first_row_per_cell_line(self, 
                                         df: pd.DataFrame):
        res = df.groupby('CELL_LINE_NAME').first().reset_index()
        return res.loc[:, ~res.columns.isin(['DRUG_ID', 'DATASET'])]
    
    def create_sparse_datasets(self):
        print(40*'=')
        print(f"{f'Creating sparse datasets...':<40}")
        print(f"{4*' '}Reading full datasets...")
        drm = pd.read_pickle(self.processed_path + 'drm_full.pkl')
        gexpr = pd.read_pickle(self.processed_path + 'gexpr_full.pkl')
        cnvg = pd.read_pickle(self.processed_path + 'cnvg_full.pkl')
        cnvp = pd.read_pickle(self.processed_path + 'cnvp_full.pkl')
        mut = pd.read_pickle(self.processed_path + 'mut_full.pkl') 
        print(f"{4*' '}Finished reading full datasets.")
        
        # We don't want any missing values for our features.
        # This would destroy the predictions of the network.
        gexpr.dropna(inplace=True)
        cnvg.dropna(inplace=True)
        cnvp.dropna(inplace=True)
        mut.dropna(inplace=True)
        
        # --------------------------------------- #
        # Find intersecting cell-lines and genes. #
        # --------------------------------------- #
        # Cell-lines.
        uniq_cl_drm = list(drm.CELL_LINE_NAME.unique())
        uniq_cl_gexpr = list(gexpr.CELL_LINE_NAME.unique())
        uniq_cl_cnvg = list(cnvg.CELL_LINE_NAME.unique())
        uniq_cl_cnvp = list(cnvp.CELL_LINE_NAME.unique())
        uniq_cl_mut = list(mut.CELL_LINE_NAME.unique()) 

        # Intersecting cell-lines between all raw datasets.
        INTER_CLS = set(uniq_cl_drm)\
            .intersection(set(uniq_cl_gexpr))\
            .intersection(set(uniq_cl_cnvg))\
            .intersection(set(uniq_cl_cnvp))\
            .intersection(set(uniq_cl_mut))

        # Genes.
        ignore = ['DATASET', 'CELL_LINE_NAME', 'DRUG_NAME', 'DRUG_ID', 
                  'SANGER_MODEL_ID', 'AUC', 'RMSE', 'Z_SCORE', 'LN_IC50']
        uniq_genes_gexpr = gexpr.columns[~gexpr.columns.isin(ignore)]
        uniq_genes_cnvg = cnvg.columns[~cnvg.columns.isin(ignore)]
        uniq_genes_cnvp = cnvp.columns[~cnvp.columns.isin(ignore)]
        uniq_genes_mut = mut.columns[~mut.columns.isin(ignore)]

        # Intersecting genes between all raw datasets.
        INTER_GENES = set(uniq_genes_gexpr)\
            .intersection(set(uniq_genes_cnvg))\
            .intersection(set(uniq_genes_cnvp))\
            .intersection(set(uniq_genes_mut))

        print(f"{4*' '}Number of intersecting cell-lines:", len(INTER_CLS))
        print(f"{4*' '}Number of intersecting genes:", len(INTER_GENES))        
        
        # Assign unique index to each of the genes.
        inter_genes_df = pd.DataFrame({'GENE_SYMBOL': list(INTER_GENES)})
        inter_genes_df['GENE_INDEX'] = inter_genes_df.index
        inter_genes_df\
            .to_csv(self.processed_path + 'sparse_inter_genes.csv', header=True, index=False)
        pd.DataFrame({'CELL_LINE_NAME': list(INTER_CLS)})\
            .to_csv(self.processed_path + 'sparse_inter_cls.csv', header=True, index=False)
        
        gexpr2 = self._get_intersecting_subset(gexpr, 'Gene Expression', INTER_GENES, INTER_CLS)
        cnvg2 = self._get_intersecting_subset(cnvg, 'CNV Gistic', INTER_GENES, INTER_CLS)
        cnvp2 = self._get_intersecting_subset(cnvp, 'CNV Picnic', INTER_GENES, INTER_CLS)
        mut2 = self._get_intersecting_subset(mut, 'Mutation', INTER_GENES, INTER_CLS)  
        
        gexpr3 = self._get_only_first_row_per_cell_line(gexpr2)
        cnvg3 = self._get_only_first_row_per_cell_line(cnvg2)
        cnvp3 = self._get_only_first_row_per_cell_line(cnvp2)
        mut3 = self._get_only_first_row_per_cell_line(mut2)
        assert gexpr3.shape == cnvg3.shape == cnvp3.shape == mut3.shape,\
            "ERROR: Not all sparsed feature datasets have the same shape."  
        print(f"{4*' '}gexpr3.shape: {gexpr3.shape}")
        
        # Save all new feature dataframes.
        gexpr3.to_pickle(self.processed_path + 'sparse_gexpr.pkl')
        cnvg3.to_pickle(self.processed_path + 'sparse_cnvg.pkl')
        cnvp3.to_pickle(self.processed_path + 'sparse_cnvp.pkl')
        mut3.to_pickle(self.processed_path + 'sparse_mut.pkl')        
    
    def create_processed_datasets(self):
        # Process drug-response matrix feature datasets.
#         self._process_gdsc_fitted()
#         self._set_landmark_genes()
#         self._process_gene_expression()
#         self._process_copy_number_variation('cnvp')
#         self._process_copy_number_variation('cnvg')
#         self._process_mutations()

        # Create sparse datasets.
        self.create_sparse_datasets()    
    
    # ---------------------------------------- #
    # PROTEIN-PROTEIN INTERATION GRAPH METHODS #
    # ---------------------------------------- #  

    def _read_protein_links(self, path: str):
        print(f"Start reading {path} ...")
        contents = gzip.open(path, "rb").read()
        data = BytesIO(contents)
        protein_links = pd.read_csv(data, sep=' ')
        print("Finished reading.")

        # Exclude the Homo Sapiens taxonomy ID from the protein columns.
        protein_links.protein1 = protein_links.protein1.str[5:]
        protein_links.protein2 = protein_links.protein2.str[5:]

        return protein_links

    def _read_protein_info(self, path: str):
        print(f"Start reading {path} ...")
        contents = gzip.open(path, "rb").read()
        data = BytesIO(contents)
        protein_info_v1 = pd.read_csv(data, sep='\t')
        print("Finished reading.")
        protein_info_v2 = protein_info_v1.rename(columns={'#string_protein_id': 'string_protein_id'}, 
                                                 inplace=False)

        # Exclude the Homo Sapiens taxonomy ID from the protein columns.
        protein_info_v2.string_protein_id = protein_info_v2.string_protein_id.str[5:]

        return protein_info_v2

    def _combine_protein_links_with_info(self,
                                         links: pd.DataFrame, 
                                         infos: pd.DataFrame):
        """
        Maps the corresponding gene symbols to the proteins in the protein link dataset.

        Args:
            links (pd.DataFrame): DataFrame containing at least the columns `protein1` and `protein2`.
            infos (pd.DataFrame): DataFrame containing at least the columns `string_protein_id` and `preferred_name`.
        Returns:
            (pd.DataFrame): DataFrame containing the corresponding `gene_symbol` per `protein1` and `protein2`.
        """
        # Get the gene symbols for the protein1 column.
        res = links.merge(right=infos[['string_protein_id', 'preferred_name']],
                          how='left',
                          left_on='protein1',
                          right_on='string_protein_id')
        res.rename(columns={'preferred_name': 'gene_symbol1'}, inplace=True)
        res.drop(['string_protein_id'], axis=1, inplace=True)

        # Get the gene symbols for the protein2 column.
        res = res.merge(right=infos[['string_protein_id', 'preferred_name']],
                        how='left',
                        left_on='protein2',
                        right_on='string_protein_id')
        res.rename(columns={'preferred_name': 'gene_symbol2'}, inplace=True)
        res.drop(['string_protein_id'], axis=1, inplace=True)

        # Drop all rows where the gene symbol has not been found.
        res.dropna(subset=['gene_symbol1', 'gene_symbol2'], inplace=True)

        assert not res[['gene_symbol1', 'gene_symbol2']].isna().sum().any(),\
            "Some gene_symbol columns are missing!"

        return res 

    def _add_gene_index(self,
                        proteins: pd.DataFrame, 
                        gene_idxs: pd.DataFrame):
        """Append corresponding index for each gene symbol column as new column(s).

        Args
            proteins (pd.DataFrame): Dataframe containing protein-protein interaction
                information. Needs to contain `gene_symbol1` and `gene_symbol2` as 
                columns.
            gene_idxs (pd.DataFrame): Datframe containing gene symbols and a 
                corresponding index. Needs to contain `GENE_SYMBOL` as a column name.
        """
        temp = proteins.merge(right=gene_idxs,
                              how='left',
                              left_on=['gene_symbol1'],
                              right_on=['GENE_SYMBOL'])
        temp.rename(columns={'GENE_INDEX': 'index_gene_symbol1'}, inplace=True)
        temp.drop(['GENE_SYMBOL'], axis=1, inplace=True)

        res = temp.merge(right=gene_idxs,
                         how='left',
                         left_on=['gene_symbol2'],
                         right_on=['GENE_SYMBOL'])
        res.rename(columns={'GENE_INDEX': 'index_gene_symbol2'}, inplace=True)
        res.drop(['GENE_SYMBOL'], axis=1, inplace=True) 

        del temp
        return res    

    def _shrink_to_only_singely_occuring_proteins(self, pinfos: pd.DataFrame):
        freq_per_gene = pinfos\
            .groupby(['preferred_name']).size()\
            .reset_index(name='freq', inplace=False)\
            .sort_values(['freq'], ascending=False)
        print(f"There are {freq_per_gene[freq_per_gene.freq>1].shape[0]} proteins with ID frequency > 2. "\
            + "They will be deleted.")

        # Remove the gene symbols which have a frequency higher than 1.
        return pinfos[~pinfos.preferred_name.isin(freq_per_gene[freq_per_gene.freq>1].preferred_name.tolist())] 

    def _get_neighbor_tuples(self,
                             proteins: pd.DataFrame, 
                             genes: pd.DataFrame):
        # Create a list of tuples which hold neighbor nodes.
        gene_symbol_tuples = list(set(list(zip(proteins.gene_symbol1, proteins.gene_symbol2))))

        # Map the list of tuples of neighbor gene nodes to the corresponding int. 
        transform_gene_tuple_to_index_tuple = lambda x : (
            genes[genes['GENE_SYMBOL']==x[0]]['GENE_INDEX'].values[0], 
            genes[genes['GENE_SYMBOL']==x[1]]['GENE_INDEX'].values[0])

        return [transform_gene_tuple_to_index_tuple(tup) for tup in gene_symbol_tuples]

    def _is_undirected_list_of_tuples(self, 
                                      neighbors: List[tuple[int]]):
        for neigh in neighbors:
            a, b = neigh[0], neigh[1]
            found = False
            for n in neighbors:
                if n == (b, a):
                    found = True
                    break
            if not found:
                print(f"ERROR: the list is directed since couldn't find ({b}, {a}) for {neigh}!")
                return False
        print("SUCCESS: The given list of tuples is undirected!")
        return True

    def _create_graph_dict_with_indices(self,
                                        proteins: pd.DataFrame, 
                                        nodes_as_indeces: List):
        """Creates a dictionary holidng as key the unique index of a gene symbol
        and as value the list of gene symbol indices which are neighbors of the 
        gene symbol index of the key."""

        # Find neighbors using the indices.
        dict_as_indices = {gene_node: [] for gene_node in nodes_as_indeces}

        # Get the edges.
        for gene in dict_as_indices:
            exists_in_gene_symbol1 = not proteins.loc[proteins.index_gene_symbol1==gene].empty
            exists_in_gene_symbol2 = not proteins.loc[proteins.index_gene_symbol2==gene].empty

            neighbor_nodes = None 
            if exists_in_gene_symbol1:
                neighbor_nodes = list(set(proteins.loc[proteins.index_gene_symbol1==gene].index_gene_symbol2))
            elif exists_in_gene_symbol2: 
                neighbor_nodes = list(set(proteins.loc[proteins.index_gene_symbol2==gene].index_gene_symbol1))
            else: 
                print(f"The gene {gene} couldn't be found in the dataset!") 

            # Set neighbors for the gene.
            dict_as_indices[gene] += [neighbor_node for neighbor_node in neighbor_nodes if neighbor_node not in dict_as_indices[gene]]

        return dict_as_indices
    
    def _get_proteins_and_genes_above_thresh(self,
                                             proteins: pd.DataFrame,
                                             genes: pd.DataFrame,
                                             thresh: float):
        """Return new proteins dataframe and gene dataframe by only selecting 
        data above the chosen threshold."""
        p_sub = proteins[proteins['combined_score'] > thresh]
        print(f"{4*' '}Choosing threshold {thresh} we have shape: {p_sub.shape}")
        print(f"{4*' '}Number of unique gene_symbol1s: {len(p_sub.gene_symbol1.unique())}")
        print(f"{4*' '}Number of unique gene_symbol2s: {len(p_sub.gene_symbol2.unique())}")        
        assert set(p_sub.gene_symbol1.unique()) == set(p_sub.gene_symbol2.unique()), \
            "ERROR: The unique gene_symbol1's are not the same as the unique gene_symbol2's!" 
        
        gene_symbols = list(set(p_sub.gene_symbol1.unique()))

        # Select only the gene symbols with a combined_score > SCORE_THRESH.
        gene_sub = genes\
            .loc[genes.GENE_SYMBOL.isin(gene_symbols)]\
            .reset_index(drop=True)
        gene_sub.loc[:, 'GENE_INDEX'] = gene_sub.index

        return p_sub, gene_sub   
    
    def _get_inter_genes_with_score_above_threshold(self, 
                                                    thresh: int,
                                                    uniq_gene_symbols: List[str],
                                                    inter_genes: pd.DataFrame) -> pd.DataFrame:       
        """This is basically the same method as _get_proteins_and_genes_above_thresh(). 
        TODO: create single one; merge both
        """
        inter_genes_above = inter_genes[inter_genes.GENE_SYMBOL.isin(uniq_gene_symbols)]\
            .reset_index(drop=True)
        inter_genes_above.loc[:, 'GENE_INDEX'] = inter_genes_above.index
        
        inter_genes_above.to_csv(
            self.processed_path + f'thresh_{thresh}_inter_genes.csv', 
            header=True, index=False
        )
        print(f"{4*' '}Created {self.processed_path + f'thresh_{thresh}_inter_genes.csv'}.")
        
        return inter_genes_above
    
    def _get_uniqs(self, 
                   df: pd.DataFrame, 
                   col: str):
        return np.unique(df[col].values).tolist()

    def _get_intersecting_cell_lines(self,
                                     gexpr, 
                                     cnvg, 
                                     cnvp, 
                                     mut):
        # TODO: add the features as *args
        # Test that all feature datasets contains exactly the same cell-lines.
        gexpr_cls = self._get_uniqs(gexpr, 'CELL_LINE_NAME')
        cnvg_cls = self._get_uniqs(cnvg, 'CELL_LINE_NAME')
        cnvp_cls = self._get_uniqs(cnvp, 'CELL_LINE_NAME')
        mut_cls = self._get_uniqs(mut, 'CELL_LINE_NAME')
        inter_cls = set(gexpr_cls) \
            .intersection(set(cnvp_cls)) \
            .intersection(set(cnvp_cls)) \
            .intersection(set(mut_cls))

        assert len(inter_cls) == len(gexpr_cls) == len(cnvg_cls) == len(cnvp_cls) == len(mut_cls), \
            "Not all feature datasets contain the exact same cell-lines as rows!"
        del gexpr_cls, cnvg_cls, cnvp_cls, mut_cls

        return inter_cls

    def _get_intersecting_genes(self, 
                                gexpr, 
                                cnvg, 
                                cnvp, 
                                mut):
        # TODO: add the features as *args        
        # Test that all feature datasets contains exactly the same gene symbols.
        inter_genes = set(np.unique(gexpr.columns.values).tolist()) \
            .intersection(set(np.unique(cnvg.columns.values).tolist())) \
            .intersection(set(np.unique(cnvp.columns.values).tolist())) \
            .intersection(set(np.unique(mut.columns.values).tolist()))

        assert len(inter_genes) == \
            len(np.unique(gexpr.columns.values).tolist()) == \
            len(np.unique(cnvg.columns.values).tolist()) == \
            len(np.unique(cnvp.columns.values).tolist()) == \
            len(np.unique(mut.columns.values).tolist()), \
                "Not all feature datasets contain the exact same gene symbols as columns!"

        return inter_genes 

    def _create_cell_line_gene_graphs(self,
                                      gene_symbols: List[str], 
                                      gene_features: List[pd.DataFrame], 
                                      cell_lines: Set[str],
                                      neighbor_gene_tuples: List[Tuple[int]]):
        """ 
        Creates pytorch geometric gene-gene interaction graphs for each of the
        given cell-lines. The each graph has the exact same topology, meaning
        the same gene symbols as nodes and the same edges. However, each graph
        has different gene feature values per node.

        Args: 
            gene_symbols (`List[str]`):
                List of gene symbols which need to be available as columns in the 
                gene features (`gene_features`).
            gene_features (`List[pd.DataFrame]`):
                List of feature pd.DataFrame's. All dataframe need to have all the 
                given cell-lines (`cls`) as indices and gene symbols (`gene_symbols`) as 
                columns.
            cls (`Set[str]`):
                List of cell-lines for which to create graphs.
            neighbor_gene_tuples (`List[Tuple[int]]`):
                List of tuples of neighbors, where the neighbors are given as gene indices, 
                not symbols.
        Returns:
            `Dict[torch_geometric.data.Data]`:
                Dictionary with the cell-line names as keys and pytorch geometric
                gene-gene interaction graphs as values.
        """        
        Gs = {}
        for cl in tqdm(cell_lines):
            # Convert the feature values to tensors and stack them up.
            cl_features = []
            for feature in gene_features:
                cl_features.append(
                    torch.tensor(feature.loc[cl][gene_symbols].values,
                                 dtype=torch.float64)
                )
            features = torch.stack(cl_features).t()

            # Generate the graph.
            edge_index = torch.tensor(neighbor_gene_tuples, dtype=torch.long).t().contiguous()
            G_cl = Data(x=features, edge_index=edge_index)

            Gs[cl] = G_cl

        return Gs   
    
    def _create_gene_feature_matrix(self,
                                    features: List[pd.DataFrame],
                                    suffixes: List[str]):
        merged = features[1]
        for i, feature in enumerate(features[1:]):
            left_suffix = suffixes[i] if i==0 else ''
            merged = pd.merge(left=merged, right=feature,
                              on=['CELL_LINE_NAME'],
                              suffixes=[left_suffix, suffixes[i+1]])
        assert merged.shape == (features[0].shape[0], 4*(features[0].shape[1]-1) + 1)
        return merged    
    
    def create_gene_gene_interaction_graph(self):        
        plinks = self._read_protein_links(self.raw_path + self.raw_protein_links_file)
        pinfos = self._read_protein_info(self.raw_path + self.raw_protein_info_file)        
        pinfos2 = self._shrink_to_only_singely_occuring_proteins(pinfos)
        proteins = self._combine_protein_links_with_info(plinks, pinfos2)  
        
        drm = pd.read_pickle(self.processed_path + 'drm_full.pkl')
        inter_genes = pd.read_csv(self.processed_path + 'sparse_inter_genes.csv')

        # Select only the rows where both gene symbols are landmark genes.
        # ----------------------------------------------------------------
        proteins2 = proteins[(proteins.gene_symbol1.isin(inter_genes.GENE_SYMBOL)) &
                             (proteins.gene_symbol2.isin(inter_genes.GENE_SYMBOL))] 
        
        # Select only the rows above the chose threshold.
        # -----------------------------------------------
        print(f"{4*' '}combined_score threshold: {self.combined_score_thresh}")
        proteins3, gene_sub = self._get_proteins_and_genes_above_thresh(proteins2, 
                                                                        inter_genes, 
                                                                        self.combined_score_thresh)

        # Select only the gene symbols with a combined_score > combined_score_thresh.
        # ---------------------------------------------------------------------------
        inter_genes_above = self._get_inter_genes_with_score_above_threshold(
            self.combined_score_thresh,
            gene_sub.GENE_SYMBOL.values.tolist(),
            inter_genes
        )

        # Add gene index to proteins dataset.
        # ----------------------------------
        proteins4 = self._add_gene_index(proteins3, 
                                         inter_genes_above)
        proteins4 = proteins4[['protein1', 'protein2', 'gene_symbol1', 'gene_symbol2', 
                               'index_gene_symbol1', 'index_gene_symbol2', 'combined_score']]

        NODES_AS_SYMBOLS = list(np.unique(proteins4[['gene_symbol1', 'gene_symbol2']].values))
        NODES_AS_INDECES = list(np.unique(proteins4[['index_gene_symbol1', 'index_gene_symbol2']].values))
        print(f"{4*' '}There will be {len(NODES_AS_SYMBOLS)} nodes in the graph.")
        print(f"{4*' '}There will be {len(NODES_AS_INDECES)} indices in the graph.")
        
        neighbor_gene_tuples = self._get_neighbor_tuples(proteins4, 
                                                         inter_genes_above)
        print(f"{4*' '}Number of neighbor tuples:", len(neighbor_gene_tuples))        
        neighbor_gene_tuples_undirected = neighbor_gene_tuples if self._is_undirected_list_of_tuples(neighbor_gene_tuples) else None      
        
        # Dictionary of neighbors per genes.
        # ---------------------------------
        dict_as_indices = self._create_graph_dict_with_indices(proteins4, 
                                                               NODES_AS_INDECES)
        
        # Load node features.
        # -------------------
        gexpr = pd.read_pickle(self.processed_path + 'sparse_gexpr.pkl')
        cnvg = pd.read_pickle(self.processed_path + 'sparse_cnvg.pkl')
        cnvp = pd.read_pickle(self.processed_path + 'sparse_cnvp.pkl')
        mut = pd.read_pickle(self.processed_path + 'sparse_mut.pkl')
        inter_cls = self._get_intersecting_cell_lines(gexpr, cnvg, cnvp, mut)
        inter_genes = self._get_intersecting_genes(gexpr, cnvg, cnvp, mut)
        print(f"{8*' '}Intersecting cell-lines: {len(inter_cls)}")
        print(f"{8*' '}Intersecting genes.    : {len(inter_genes)}")        
  
        
#         # Build gene-gene interaction graph.
#         # ----------------------------------
#         gexpr.set_index('CELL_LINE_NAME', inplace=True)
#         cnvg.set_index('CELL_LINE_NAME', inplace=True)
#         cnvp.set_index('CELL_LINE_NAME', inplace=True)
#         mut.set_index('CELL_LINE_NAME', inplace=True) 
#         print(f"{4*' '}Creating gene-gene interaction graph...")
#         cl_graphs = self._create_cell_line_gene_graphs(
#             NODES_AS_SYMBOLS, 
#             [
#                 gexpr, 
#                 cnvg, 
#                 cnvp, 
#                 mut
#             ],
#             inter_cls,
#             neighbor_gene_tuples_undirected) 
#         print(f"{4*' '}Finished creating gene-gene interaction graph.")
#         # Showcase topology for some cell-line examples.
#         for cl in gexpr.index[:5].tolist():
#             print(f"{8*' '}Cell-line: {cl:8s}   Graph: {cl_graphs[cl]}")        
        
#         with open(self.processed_path + f'thresh_{self.combined_score_thresh}_gene_graphs.pkl', 'wb') as f:
#             pickle.dump(cl_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)  
#         print(f"Successfully saved full gene-gene graphs in {self.processed_path + f'thresh_{self.combined_score_thresh}_gene_graphs.pkl'}.")
            
        # Feature subset with only the genes over the chosen threshold.
        # -------------------------------------------------------------
#         gexpr.reset_index(inplace=True)
#         cnvg.reset_index(inplace=True)
#         cnvp.reset_index(inplace=True)
#         mut.reset_index(inplace=True)    
        
        gexpr2 = gexpr.loc[:, gexpr.columns.isin(['CELL_LINE_NAME'] + inter_genes_above.GENE_SYMBOL.values.tolist())]
        cnvg2 = cnvg.loc[:, cnvg.columns.isin(['CELL_LINE_NAME'] + inter_genes_above.GENE_SYMBOL.values.tolist())]
        cnvp2 = cnvp.loc[:, cnvp.columns.isin(['CELL_LINE_NAME'] + inter_genes_above.GENE_SYMBOL.values.tolist())]
        mut2 = mut.loc[:, mut.columns.isin(['CELL_LINE_NAME'] + inter_genes_above.GENE_SYMBOL.values.tolist())]
        assert gexpr2.shape == cnvg2.shape == cnvp2.shape == mut2.shape, \
            "ERROR: The shapes of all feature dataframes are not equal." 
        print(f"{4*' '}Each new feature dataset has shape: {gexpr2.shape}")
        
#         # These dataframe have been sparsed by intersecting genes after chosing 
#         # only the genes with combined_score > 700.
#         gexpr2.to_pickle(self.processed_path + f'thresh_{combined_score_thresh}_gexpr.pkl')
#         cnvg2.to_pickle(self.processed_path + f'thresh_{combined_score_thresh}_cnvg.pkl')
#         cnvp2.to_pickle(self.processed_path + f'thresh_{combined_score_thresh}_cnvp.pkl')
#         mut2.to_pickle(self.processed_path + f'thresh_{combined_score_thresh}_mut.pkl')        
        
        assert self.gdsc.upper() in list(drm.DATASET.unique()),\
            f"ERROR: The chosen GDSC database {self.gdsc.upper()} is not in the given drug response matrix. "\
            + f"Only {list(drm.DATASET.unique())} are available."

        inter_cls_gdsc = set(drm[drm.DATASET==self.gdsc.upper()].CELL_LINE_NAME.unique())\
            .intersection(set(gexpr2.CELL_LINE_NAME.unique()))\
            .intersection(set(cnvg2.CELL_LINE_NAME.unique()))\
            .intersection(set(cnvp2.CELL_LINE_NAME.unique()))\
            .intersection(set(mut2.CELL_LINE_NAME.unique()))

        print(f"Since GDSC {self.gdsc[-1]} database was chosen the number of intersecting cell-lines is {len(list(inter_cls_gdsc))}")
        
        pd.DataFrame({'CELL_LINE_NAME': list(inter_cls_gdsc)})\
            .to_csv(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_inter_cls.csv', 
                    header=True, index=False) 
        
        drmGDSC = drm[drm.DATASET==self.gdsc.upper()]
        drmGDSC = drmGDSC[drmGDSC.CELL_LINE_NAME.isin(inter_cls_gdsc)]
        gexprGDSC = gexpr2[gexpr2.CELL_LINE_NAME.isin(inter_cls_gdsc)]
        cnvgGDSC = cnvg2[cnvg2.CELL_LINE_NAME.isin(inter_cls_gdsc)]
        cnvpGDSC = cnvp2[cnvp2.CELL_LINE_NAME.isin(inter_cls_gdsc)]
        mutGDSC = mut2[mut2.CELL_LINE_NAME.isin(inter_cls_gdsc)] 

        # Sparsed by combined score > 700 and intersecting cell-lines for only GDSC2
        gexprGDSC.to_pickle(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_gexpr.pkl')
        cnvgGDSC.to_pickle(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_cnvg.pkl')
        cnvpGDSC.to_pickle(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_cnvp.pkl')
        mutGDSC.to_pickle(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_mut.pkl') 
        
        # As graph.
        # ---------
        gexprGDSC.set_index('CELL_LINE_NAME', inplace=True)
        cnvgGDSC.set_index('CELL_LINE_NAME', inplace=True)
        cnvpGDSC.set_index('CELL_LINE_NAME', inplace=True)
        mutGDSC.set_index('CELL_LINE_NAME', inplace=True)  
        print(f"{4*' '}Creating gene-gene interaction graph for {self.gdsc}...")        
        cl_graphs_GDSC = self._create_cell_line_gene_graphs(
            inter_genes_above.GENE_SYMBOL.values.tolist(), 
            [
                gexprGDSC, 
                cnvgGDSC, 
                cnvpGDSC, 
                mutGDSC
            ],
            inter_cls_gdsc,
            neighbor_gene_tuples_undirected
        )         
        print(f"{4*' '}Finished creating gene-gene interaction graph for {self.gdsc}.")
        # Showcase topology for some cell-line examples.
        for cl in gexprGDSC.index[:5].tolist():
            print(f"{8*' '}Cell-line: {cl:8s}   Graph: {cl_graphs_GDSC[cl]}") 
            
        with open(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_gene_graphs.pkl', 'wb') as f:
            pickle.dump(cl_graphs_GDSC, f, protocol=pickle.HIGHEST_PROTOCOL) 
        print(f"Successfully saved full gene-gene graphs for {self.gdsc} in {self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_gene_graphs.pkl'}.")            
        
        # As table.
        # ---------
        gexprGDSC.reset_index(inplace=True)
        cnvgGDSC.reset_index(inplace=True)
        cnvpGDSC.reset_index(inplace=True)
        mutGDSC.reset_index(inplace=True)         
        merged = self._create_gene_feature_matrix([gexprGDSC, cnvgGDSC, cnvpGDSC, mutGDSC],
                                                  ['_gexpr', '_cnvg', '_cnvp', '_mut'])
        print(f"{4*' '}Created gene feature matrices with shape : {merged.shape}")
        
        merged = pd.concat([gexprGDSC.set_index('CELL_LINE_NAME'), 
                            cnvgGDSC.set_index('CELL_LINE_NAME'),
                            cnvpGDSC.set_index('CELL_LINE_NAME'),
                            mutGDSC.set_index('CELL_LINE_NAME')], 
                           keys=('gexpr', 'cnvg', 'cnvp', 'mut'), 
                           axis=1)
        merged.columns = merged.columns.map(lambda x: f"{x[1]}_{x[0]}")
        assert len([col for col in merged.columns if '_gexpr' in col]) == \
               len([col for col in merged.columns if '_cnvg' in col]) == \
               len([col for col in merged.columns if '_cnvp' in col]) == \
               len([col for col in merged.columns if '_mut' in col]), \
            "ERROR: Number of gene columns per feature is not equal."
        merged.reset_index(inplace=True) 
        print(f"{8*' '}Number of      genes: {len([col for col in merged.columns if '_gexpr' in col])}")  
        print(f"{8*' '}Number of cell-lines: {len(merged.CELL_LINE_NAME.unique())}")        
        
        merged.to_pickle(self.gdsc_thresh_path + f"thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_gene_mat.pkl") 
        print(f"Successfully saved full gene-gene matrix for {self.gdsc.upper()} in {self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_gene_mat.pkl'}.")                 
        
        
    # ---------------------------- #
    # DRUG SMILES GRAPH AND MATRIX #
    # ---------------------------- #          
    def _get_demorgen_fingerprints(self,
                                   drugs: pd.Series,
                                   radius: int = 2,
                                   n_bits: int = 256,
                                   path_drug_smiles: str = "data/GDSC/GDSC_compounds_inchi_key_with_smiles.csv"):
        """

        :param drugs: pandas Series of GDSC drug names for which to query the fingerprints
        :param radius: see rdkit documentation GetMorganFingerprintAsBitVect
        :param n_bits: length of the fingerprint
        :param path_drug_smiles: path to .csv file containing "smiles" and "drug_name" column
        :return: List of fingerprints or nans, if fingerprint inaccessible
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Load smiles data for the GDSC compounds.
        GDSC_compound_identification = pd.read_csv(path_drug_smiles, index_col=0)
        GDSC_compound_identification = GDSC_compound_identification.drop_duplicates(
            "drug_name"
        )
        GDSC_compound_identification = GDSC_compound_identification.loc[
            GDSC_compound_identification.smiles != "not_found"
        ]

        # Convert smiles to RDkit mols.
        compound_name_to_rdkit_molecule_map = dict()
        for compound_name, compound_smiles in zip(
            GDSC_compound_identification.drug_name, GDSC_compound_identification.smiles
        ):
            compound_name_to_rdkit_molecule_map[compound_name] = Chem.MolFromSmiles(
                compound_smiles
            )

        # Convert RDkit mols to demorgan fingerprints.
        compound_name_to_demorgan_fingerprint = dict()
        for compound_name in GDSC_compound_identification.drug_name:
            mol = compound_name_to_rdkit_molecule_map[compound_name]
            if mol is not None:
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=n_bits
                )
                compound_name_to_demorgan_fingerprint[compound_name] = fingerprint
            else:
                compound_name_to_demorgan_fingerprint[compound_name] = None

        # Apply conversion to input drugs.
        fingerprints = []
        for drug in drugs:
            if drug in compound_name_to_demorgan_fingerprint:
                fingerprints.append(np.array(compound_name_to_demorgan_fingerprint[drug]))
            else:
                fingerprints.append(None)

        return fingerprints  
    
    def create_drug_datasets(self):
        # As table.
        # ---------
        drm = pd.read_pickle(self.processed_path + 'drm_full.pkl')
        cell_lines = pd.read_csv(self.gdsc_thresh_path + f'thresh_{self.gdsc.lower()}_{self.combined_score_thresh}_inter_cls.csv')
        drmGDSC = drm[drm.DATASET==self.gdsc.upper()]
        drmGDSC = drmGDSC[drmGDSC.CELL_LINE_NAME.isin(cell_lines.CELL_LINE_NAME.values.tolist())]       
        uniq_drug_names = list(drmGDSC.DRUG_NAME.unique())
        print(f"{4*' '}Number of unique cell-lines:", len(drmGDSC.CELL_LINE_NAME.unique()))
        print(f"{4*' '}Number of unique DRUG_NAMEs:", len(uniq_drug_names))         

        smiles = pd.read_csv(self.raw_path + self.raw_smiles_file, index_col=0)
        
        N_BITS = 256

        # Returns a list of fingerprints or nans.
        fps = self._get_demorgen_fingerprints(drugs=uniq_drug_names,
                                              n_bits=N_BITS,
                                              path_drug_smiles=self.raw_path + self.raw_smiles_file)

        # Append the fingerprints to the corresponding drug names. 
        drug_name_fps_full = {uniq_drug_name: fps[i] for i, uniq_drug_name in enumerate(uniq_drug_names)}
        drug_name_fps = {drug_name: fp for drug_name, fp in drug_name_fps_full.items() if fp is not None}
        non_drugs = list(set(drug_name_fps_full.keys()).difference(set(drug_name_fps.keys())))        
        print(f"{4*' '}Number of drugs: {len(drug_name_fps_full.keys())}")        
        print(f"{4*' '}Number of drugs with not None fingerprint: {len(drug_name_fps.keys())}")
        print(f"{4*' '}Number of drugs which have a None fingerprint: {len(non_drugs)}")            
    
#         drmGDSC_v2 = drmGDSC_v2[~drmGDSC_v2.DRUG_NAME.isin(non_drugs)]
        drmGDSC_v2 = drmGDSC[~drmGDSC.DRUG_NAME.isin(non_drugs)]
        assert len(drmGDSC.CELL_LINE_NAME.unique()) == len(drmGDSC_v2.CELL_LINE_NAME.unique()), \
            "Some cell-line have been removed because they have only drugs which have a None fingerprint."  
        
        # Remove the DRUG_NAME's which have more then one corresponding DRUG_ID.
        non_uniqs = drmGDSC_v2[['DRUG_ID', 'DRUG_NAME']]\
            .groupby(['DRUG_NAME']).nunique()\
            .sort_values(['DRUG_ID'], ascending=False)\
            .reset_index().rename(columns={'DRUG_ID': 'count'})
        non_uniqs = non_uniqs[non_uniqs['count'] > 1]
        print(f"{4*' '}Number of DRUG_NAME's which have more then 1 DRUG_ID:", non_uniqs.shape[0])
        non_uniq_drug_names = non_uniqs.DRUG_NAME.tolist()
        print(non_uniq_drug_names)

        # Remove these drug names from the drug response matrix and the smiles matrix.
        drmGDSC_v3 = drmGDSC_v2[~drmGDSC_v2.DRUG_NAME.isin(non_uniq_drug_names)]
        print(drmGDSC_v2.shape)
        print(drmGDSC_v3.shape)
        print(f"{4*' '}Number of unique cell-lines before:", len(drmGDSC_v2.CELL_LINE_NAME.unique()))
        print(f"{4*' '}Number of unique cell-lines after:", len(drmGDSC_v3.CELL_LINE_NAME.unique()))
        print(f"{4*' '}Number of unique drug names:", len(drmGDSC_v3.DRUG_NAME.unique()))
        print(f"{4*' '}Number of unique drug id:", len(drmGDSC_v3.DRUG_ID.unique()))

        drug_name_fps_v2 = drug_name_fps
        for drug_name in non_uniq_drug_names:
            drug_name_fps_v2.pop(drug_name, None)
        print(f"{4*' '}Number of drug name keys:", len(drug_name_fps_v2.keys()))

        assert len(drmGDSC_v3.DRUG_NAME.unique()) == len(drmGDSC_v3.DRUG_ID.unique()) == len(drug_name_fps_v2.keys()), \
            "ERROR: There is some mismatch in the DRUG_NAME's and DRUG_ID's between the drug response matrix and drug smiles dictionary."        
        
        drug_name_fps_df = pd.DataFrame\
            .from_dict(drug_name_fps_v2, orient='index')\
            .rename_axis('DRUG_NAME')\
            .reset_index()
        
        drug_id_fps_df = pd.merge(left=drug_name_fps_df, 
                                  right=drmGDSC[['DRUG_NAME', 'DRUG_ID']], 
                                  how='left', on=['DRUG_NAME']).drop_duplicates()
        drug_id_fps_df.reset_index(inplace=True, drop=True)
        drug_id_fps_df.insert(1, 'DRUG_ID', drug_id_fps_df.pop('DRUG_ID'))        
        
        drug_id_fps_df = drug_id_fps_df\
            .loc[:, ~drug_id_fps_df.columns.isin(['DRUG_NAME'])]

        drug_id_fps_dict = drug_id_fps_df\
            .loc[:, ~drug_id_fps_df.columns.isin(['DRUG_NAME'])]\
            .set_index('DRUG_ID').T.to_dict('list')   
    
        # Save the DRUG_NAME - fingerprint dictionary to a file.
        with open(self.gdsc_path + f'{self.gdsc.lower()}_smiles_dict.pkl', 'wb') as f:
            pickle.dump(drug_id_fps_dict, f) # As dictionary.
        drug_id_fps_df.to_pickle(self.gdsc_path + f'{self.gdsc.lower()}_smiles_mat.pkl') # As matrix.
        print(f"Successfully saved full SMILES matrix for {self.gdsc} in {self.gdsc_path + f'{self.gdsc.lower()}_smiles_mat.pkl'}.")  
        
        # Save the new drug response matrix with only DRUG_ID's which have a fingerprint.
        drmGDSC_v3.to_pickle(self.gdsc_path + f'{self.gdsc.lower()}_drm.pkl')    
        print(f"Successfully saved new drug response matrix which has FP for each row for {self.gdsc} in {self.gdsc_path + f'{self.gdsc.lower()}_drm.pkl'}.")    
        
        # As graph.
        # ---------
        assert drug_name_fps_df.shape[0] == len(drmGDSC_v3.DRUG_NAME.unique()) == len(drmGDSC_v3.DRUG_ID.unique()),\
            "ERROR: mismatch in the number of unique DRUG_NAME's."
        
        # Join the fingerprints to get also the DRUG_ID's.
        smiles = pd.merge(left=drug_name_fps_df['DRUG_NAME'], 
                          right=drmGDSC_v3[['DRUG_NAME', 'DRUG_ID']], 
                          how='left', on=['DRUG_NAME']).drop_duplicates()
        smiles.reset_index(inplace=True, drop=True)
        # Root smiles dataframe with the SMILES string.
        smiles_root = pd.read_csv(self.raw_path + self.raw_smiles_file, index_col=0)

        # Join the SMILES string onto the subset.
        smiles2 = pd.merge(left=smiles,
                           right=smiles_root[['drug_name', 'smiles']],
                           left_on=['DRUG_NAME'],
                           right_on=['drug_name'],
                           how='left')[['DRUG_NAME', 'DRUG_ID', 'smiles']]\
                    .rename(columns={'smiles': 'SMILES'})\
                    .drop_duplicates()
        smiles2 = smiles2[smiles2.SMILES != 'not_found']        
        
        smiles2[['DRUG_NAME', 'DRUG_ID']].to_csv(
            self.gdsc_path + f'{self.gdsc.lower()}_drug_name_id_map.csv', 
            header=True, index=False)
        print(f"Successfully saved {self.gdsc_path + f'{self.gdsc.lower()}_drug_name_id_map.csv'}.")       

        # Create dictionary with DRUG_ID as key and smiles molecular graph as value.
        smiles_graphs = {}
        for i in range(smiles2.shape[0]):
            drug_name, drug_id, smiles = smiles2.iloc[i]
            smiles_graphs[drug_id] = from_smiles(smiles)

        print(f"{4*' '}Number of keys/drugs : {len(smiles_graphs.keys())}")
        # Print some examples.
        for i in range(5):
            print(f"{4*' '}drug_id: {smiles2.iloc[i].DRUG_ID:5.0f} | drug_name: {smiles2.iloc[i].DRUG_NAME:15s} | graph: {smiles_graphs[smiles2.iloc[i].DRUG_ID]}")        
        
        with open(self.gdsc_path + f'{self.gdsc.lower()}_smiles_graphs.pkl', 'wb') as f:
            pickle.dump(smiles_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved SMILES graphs in {self.gdsc_path + f'{self.gdsc.lower()}_smiles_graphs.pkl'}.")       
        
    
    # Create final datasets.
#         self.create_final_training_datasets(combined_score_thresh, gdsc_db)
        
    
        