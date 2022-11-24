
# Datasets

For the creation of training datasets multiple resources have been used. All of the mentioned datasets are open-source and free for download.

## Final Datasets
The following datasets have been created:

| Dataset | Description | Used For | Code | Folder |
| ------- | ----------- | -------- | ---- | ------ | 
| `drug_response_matrix.pkl` | Drug response matrix holding the `LN_IC50` scores for cell-line drug tuples. | `TabTab`, `TabGraph`, `GraphTab`, `GraphGraph` | TODO | TODO |
| `cl_mat.pkl` | Cell-line gene matrix holding the gene features as columns for the cell-line. | `TabTab`, `TabGraph` | TODO | TODO |
| `cl_graphs.pkl` | Cell-line gene-gene interaction graphs holding the gene-gene interaction graphs for each cell-line. | `GraphTab`, `GraphGraph` | TODO | TODO |
| `drug_mat.pkl` | Drug matrix holding the SMILES fingerprints as columns for the drug. | `TabTab`, `GraphTab` | TODO | TODO |
| `drug_graphs.pkl` | SMILES fingerprint graphs for each drug. | `TabGraph`, `GraphGraph` | TODO | TODO | 

<details><summary><ins>Additional helper datasets:</ins></summary>
<p>

| Dataset | Description | Code | Folder |
| ------- | ----------- | ---- | ------ |
| `gene_expr.pkl` | Table holding gene expression values for genes. | TODO | TODO |
| `cnv_gistic.pkl` | Table holding copy number variations for genes. | TODO | TODO |
| `cnv_picnic.pkl` | Table holding copy number variations for genes. | TODO | TODO |
| `mut.pkl` | Table holding mutational information for genes. | TODO | TODO |
| `gene_symbol_idx_map.csv` | Table holding the used gene symbols and their corresponding created gene index. | TODO | TODO |
| `landmark_genes.csv` | List of landmark genes to select only a subset of all genes in the GDSC tables. | TODO | TODO |
| `GDSC_compounds_inchi_key_with_smiles.csv` | SMILES fingerprints for drugs. | TODO | TODO |

</p>
</details>


### Dataset Preview
To get a rough overview of how the final datasets are looking like, the following shows the important statistics for each:

#### <ins>`drug_response_matrix.pkl`</ins>

#### <ins>`cl_mat.pkl`</ins>

#### <ins>`cl_graphs.pkl`</ins>

#### <ins>`drug_mat.pkl`</ins>

#### <ins>`drug_graphs.pkl`</ins>


## Root Datasets
---
Mainly three resources have been used to create the datasets used for modeling. 

1. GDSC tables, which provide existing experimental results (`LN_IC50` values) for a set of cell-line drug tuples. In this project these tables are being used to create the drug response matrix.
2. Cell feature tables, which provide gene feature information for a set of cell-lines. In this project these tables are being used to obtain numerical features for cell-lines which will create the cell-line gene matrix and corresponding cell-line gene-gene interaction graph.
3. Link interaction tables, which provide information about the relation ship of a set of protein-protein tuples. These protein-protein interaction are being used to map to their corresponding gene-gene tuples to create the gene-gene interaction graphs.

The following subsection will provide information about these root datasets.

---
### GDSC Base
#### <ins>__Drug Screening - IC50s__:</ins>

- Origin: https://www.cancerrxgene.org/downloads/bulk_download 
  - Screening Data > Drug Screening - IC50s
- Datasets: 
  - [GDSC1-dataset](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_fitted_dose_response_25Feb20.xlsx): `GDSC1_fitted_dose_response_25Feb20.xlsx` (27,3 MB)
  - [GDSC2-dataset](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC2_fitted_dose_response_25Feb20.xlsx): `GDSC2_fitted_dose_response_25Feb20.xlsx` (11,9 MB)

<details><summary>Click to see column descriptions:</summary>
<p>

| Column | Description |
| ------ | ----------- | 
| `DATASET` |  Name of the dataset. |
| `NLME_RESULT_ID` |  |
| `NLME_CURVE_ID` |  |
| `COSMIC_ID` | Cell identifier from the COSMIC database. |
| `CELL_LINE_NAME` | Primary name for the cell line. |
| `SANGER_MODEL_ID` |  |
| `TCGA_DESC` |  |
| `DRUG_ID` | Unique identifier for a drug. Used for internal lab tracking. |
| `DRUG_NAME` | Primary name for the drug. |
| `PUTATIVE_TARGET` | Putative drug target. |
| `PATHWAY_NAME` |  |
| `COMPANY_ID` |  |
| `WEBRELEASE` |  |
| `MIN_CONC` | Minimum screening concentration of the drug. |
| `MAX_CONC` | Maximum screening concentration of the drug. |
| `LN_IC50` | Natural log of the fitted IC50. To convert to micromolar take the exponent of this value, i.e. $\exp(ln(IC50))$. |
| `AUC` | Area Under the Curve for the fitted model. Presented as a fraction of the total area between the highest and lowest screening concentration. |
| `RMSE` | Root Mean Squared Error, a measurement of how well the modelled curve fits the data points. |
| `Z_SCORE` | Z score of the $ln(IC50)$ ($x$) comparing it to the mean ($\mu$) and standard deviation ($\sigma^2$) of the LN_IC50 values for the drug in question over all cell lines treated. $Z = \frac{x-\mu}{\sigma^2}$ |

</p>
</details>

#### <ins>__Drug Screening - Raw data__:</ins> 
- Origin: https://www.cancerrxgene.org/downloads/bulk_download
  - Screening Data > Drug Screening - Raw data
- Datasets: 
  - [GDSC1-raw-data](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_public_raw_data_25Feb20.csv): `GDSC1_public_raw_data_25Feb20.csv` (701,9 MB)
    - Shape: (5,837,703 , 18)
  - [GDSC2-raw-data](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC2_public_raw_data_25Feb20.csv): `GDSC2_public_raw_data_25Feb20.csv` (865,8 MB)
    - Shape: (6,646,430 , 18)

<details>
  <summary>Click to see column descriptions:</summary>

| Column | Description |
| ------ | ----------- | 
| `RESEARCH_PROJECT` | Project name for the dataset. |
| `BARCODE` | Unique barcode for screening assay plate |
| `SCAN_ID` | Unique id for the scan of the plate by the plate reader - fluorescence measurement data. A plate might be scanned more than once but only one `SCAN_ID` will pass internal QC. Therefore there is a one to one correspondence between `BARCODE` and `SCAN_ID` in the published data. |
| `DATE_CREATED` | Date that the plate was seeded with cell line. |
| `SCAN_DATE` | Date the experiment finished and measurement was taken (scanning). |
| `CELL_ID` | Unique GDSC identifier for the cell line expansion seeded on the plate. Each time a cell line is expanded from frozen stocks it is assigned a new `CELL_ID`. | 
| `MASTER_CELL_ID` | Unique GDSC identifier for the cell line seeded on the plate. A particular cell line will have a single `MASTER_CELL_ID` but can have multiple `CELL_ID`. | 
| `COSMIC_ID` | Identifier of the cell line in the COSMIC database if available. There is a one to one correspondence between `MASTER_CELL_ID` and `COSMIC_ID`. | 
| `CELL_LINE_NAME` | Name of the plated cell line. Again this will have a one to one correspondence with `MASTER_CELL_ID`. | 
| `SEEDING_DENSITY` | Number of cells seeded per well of screening plate. This number is the same for all wells on a plate. | 
| `DRUGSET_ID` | The set of drugs used to treat the plate and the associated plate layout. | 
| `ASSAY` | End point assay type used to assess cell viability, e.g., Glo is Promega CellTiter-Glo. | 
| `DURATION` | Duration of the assay in days from cell line drug treatment to end point measurement. | 
| `POSITION` | Plate well position numbered row-wise. 1536 well plates have 48 columns and 384 well plates have 24. | 
| `TAG` | Label to identify well treatment - see description below. It is possible to have more than one tag per well `POSITION` such that in the raw data files (csv) there may be more than one row per plate well position, e.g., L12-D1-S + DMSO. | 
| `DRUG_ID` | Unique identifier for the drug used for treatment. In the absence of a drug treatment, e.g., a negative control this field will be NA. | 
| `CONC` | Micromolar concentration of the drug id used for treatment. As with `DRUG_ID` this field can be NA. | 
| `INTENSITY` | Fluorescence measurement at the end of the assay. The fluorescence is a result of `ASSAY` and is an indicator of cell viability. | 

</details>

---
### Cell Features

#### <ins>__Gene Expression__:</ins>
- Origin: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html 
  - Omic: EXP > DataType: Preprocessed > Objects: Cell-lines > Keywords: RMA normalised expression data for cell-lines > Dataset
- Dataset: [Cell_line_RMA_proc_basalExp.txt.zip](https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip)
  - `Cell_line_RMA_proc_basalExp.txt` (306.3 MB)

#### <ins>__Copy Number Variation__:</ins>
- Origin: https://cellmodelpassports.sanger.ac.uk/downloads 
  - Copy Number Data > Copy Number (SNP6) > Downloiad SNP6 CNV data (15.85 MB)
- Dataset: [cnv_20191101.zip](https://cog.sanger.ac.uk/cmp/download/cnv_20191101.zip)
  - `cnv_abs_copy_number_picnic_20191101.csv` (97 MB)
  - `cnv_gistic_20191101.csv` (86.7 MB)
  - Additional Description: "PICNIC absolute copy numbers and GISTIC scores derived from Affymetrix SNP6.0 array data."

#### <ins>__Mutations__:</ins>
- Origin: https://cellmodelpassports.sanger.ac.uk/downloads 
  - Mutation Data > Mutations All > Download all mutations (94.8 MB)
  - Mutation Data > Mutations Summary > Download driver mutations (288.08 kB)
- Dataset: [mutations_all_20220315.zip](https://cog.sanger.ac.uk/cmp/download/mutations_all_20220315.zip)
  - `mutations_all_20220315.csv` (749.6 MB) 
    - Shape: (8,322,616 , 13)
  - Additional Description: "A list of all mutations present in all sequenced models." 
- Dataset: [mutations_summary_20220315.zip](https://cog.sanger.ac.uk/cmp/download/mutations_summary_20220509.zip)
  - `mutations_summary_20220315.csv` (1.2 MB) 
    - Shape: (11,609 , 13)
  - Additional Description: "A list of cancer driver mutations present in all sequenced models."

---
### Link Interaction Data

#### <ins>__Protein Links:__</ins>
- Origin: https://string-db.org/cgi/download?sessionId=bUA6KHr2nITv&species_text=Homo+sapiens 
  - Search "Homo Sapiens" > Interaction Data
- Dataset: [9606.protein.links.detailed.v11.5.txt.gz](https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz)
  - `9606.protein.links.detailed.v11.5.txt` (115.5 MB)
    - Shape: (11,938,498 , 10)
  - Additional Description: "protein network data (full network, incl. subscores per channel)"

neighborhood	fusion	cooccurence	coexpression	experimental	database	textmining	combined_score

<details>
  <summary>Click to see column descriptions:</summary>

| Column | Range | Description |
| ------ | ----- | ----------- | 
| `protein1` | | Internal identifier of protein A. |
| `protein2` | | Internal identifier of protein B. |
| `neighborhood` | $\in [0, 385]$ | |
| `fusion` | $\in [0, 900]$ | |
| `cooccurence` | $\in [0, 448]$ | |
| `coexpression` | $\in [0, 999]$ | |
| `experimental` | $\in [0, 999]$ | |
| `database` | $\in [0, 900]$ | |
| `textmining` | $\in [0, 997]$ | |
| `combined_score` | $\in [150, 999]$ | The combined score of all evidence scores (including transferred scores). Lies $\in [0, 1,000]$. |

More descriptions can be found under https://string-db.org/cgi/help?sessionId=bUA6KHr2nITv . 

</details>

#### <ins>__Protein Info:__</ins>
- Origin: https://string-db.org/cgi/download?sessionId=bUA6KHr2nITv&species_text=Homo+sapiens
  - Search "Homo Sapiens" > Interaction Data
- Dataset: [9606.protein.info.v11.5.txt.gz](https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz)
  - `9606.protein.info.v11.5.txt` (1.9 MB)
    - Shape: (19,566 , 4)
  - Additional Description: "list of STRING proteins incl. their display names and descriptions"

<details>
  <summary>Click to see column descriptions:</summary>

| Column              | Description | Example |
| ------------------- | ----------- | ------- |
| `string_protein_id` | Protein identifier | `9606.ENSP00000000233	` |
| `preferred_name`    | Gene identifier. Same as `GENE_SYMBOL`. | `M6PR` |
| `protein_size`      | Size of the protein in aa (amino acids)| `277` |
| `annotation`        | Full name of the protein. | `Cation-dependent mannose-6-phosphate receptor;...` |

- The `string_protein_id` protein is encoded by the `preferred_name` gene
- The `preferred_name` gene encodes the `string_protein_id` protein

</details>

# Todos

- [ ] provide links in the dataset summary table for each approach and to each folder