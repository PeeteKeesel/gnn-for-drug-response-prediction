

__Drug Screening - IC50s__: 

- Origin: https://www.cancerrxgene.org/downloads/bulk_download 
- Datasets: 
  - [GDSC1-dataset](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_fitted_dose_response_25Feb20.xlsx): `GDSC1_fitted_dose_response_25Feb20.xlsx`
  - [GDSC2-dataset](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC2_fitted_dose_response_25Feb20.xlsx): `GDSC2_fitted_dose_response_25Feb20.xlsx`

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
| `LN_IC50` | Natural log of the fitted IC50. To convert to micromolar take the exponent of this value, i.e. $\exp(\text{IC50\_nat\_log})$. |
| `AUC` | Area Under the Curve for the fitted model. Presented as a fraction of the total area between the highest and lowest screening concentration. |
| `RMSE` | Root Mean Squared Error, a measurement of how well the modelled curve fits the data points. |
| `Z_SCORE` | Z score of the LN_IC50 ($x$) comparing it to the mean ($\mu$) and standard deviation ($\sigma^2$) of the LN_IC50 values for the drug in question over all cell lines treated. $Z = \frac{x-\mu}{\sigma^2}$ |

</p>
</details>

__Drug Screening - Raw data__: 
- Origin: https://www.cancerrxgene.org/downloads/bulk_download 
- Datasets: 
  - [GDSC1-raw-data](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_public_raw_data_25Feb20.csv): `GDSC1_public_raw_data_25Feb20.csv`
  - [GDSC2-raw-data](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC2_public_raw_data_25Feb20.csv): `GDSC2_public_raw_data_25Feb20.csv`

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