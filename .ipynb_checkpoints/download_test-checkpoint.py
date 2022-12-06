from processor import Processor


class Downloader(Enum):
    GDSC1 = 'ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC1_fitted_dose_response_24Jul22.xlsx'
    GDSC2 = 'ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC2_fitted_dose_response_24Jul22.xlsx'
    GEXPR = 'https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip'
    CL_DETAILS = 'ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/Cell_Lines_Details.xlsx'  
    CNV = 'https://cog.sanger.ac.uk/cmp/download/cnv_20191101.zip'
    MUT = 'https://cog.sanger.ac.uk/cmp/download/mutations_all_20220315.zip'
    PROTEIN_LINKS = 'https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz'
    PROTEIN_INFO = 'https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz'

    # TODO: find from where the below are coming?    
    # DRUG = 'GDSC_compounds_inchi_key_with_smiles.csv'
    # LANDMARK_GENES = 'landmark_genes.csv'
    
    @classmethod
    def get_names(cls):
        return [data.name for data in cls]
    

class Downloader(Processor):
    def __init__(self, 
                 raw_path: str,
                 download_links=DownloadLinks):
        super(Processor, self).__init__(raw_path=raw_path)
        self.download_links = download_links

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
        
     