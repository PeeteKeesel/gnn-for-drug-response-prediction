import random
import logging
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

from argparse                      import ArgumentParser
from pathlib                       import Path
from config                        import PATH_SUMMARY_DATASETS
from src.models.TabTab.tab_tab     import TabTabDataset, create_tab_tab_datasets, BuildTabTabModel, TabTab_v1
from src.models.GraphTab.graph_tab import GraphTabDataset, create_graph_tab_datasets, BuildGraphTabModel, GraphTab_v1
from src.models.TabGraph.tab_graph import TabGraphDataset, create_tab_graph_datasets, BuildTabGraphModel, TabGraph_v1
from src.preprocess.processor      import Processor
from sklearn.model_selection       import train_test_split
from torch_geometric.loader        import DataLoader

PERFORMANCES = 'performances/'


def parse_args():
    parser = ArgumentParser(description='GNNs for Drug Response Prediction in Cancer')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=1_000, 
                        help='the batch size (default: 10)')
    parser.add_argument('--lr', type=int, default=0.0001, 
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.5, 
                        help='validation set ratio inside the test set (default: 0.5)')
    parser.add_argument('--num_epochs', type=int, default=2, 
                        help='number of epochs (default: )')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='number of workers for DataLoader (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='dropout probability (default: 0.1)')
    parser.add_argument('--model', type=str, default='GraphTab', 
                        help='name of the model to run, options: ' + \
                             '[`TabTab`, `GraphTab`, `TabGraph`, `GraphGraph`,' + \
                             ' `tabtab`, `graphtab`, `tabgraph`, `graphgraph`, ' + \
                             ' `TT`, `GT`, `TG`, `GG`, `tt`, `gt`, `tg`, `gg` ]')
    parser.add_argument('--version', type=str, default='v3', 
                        help='model version to run') # TODO: this is not used currently
    parser.add_argument('--download', type=str, default='n', 
                        help="If raw data should be downloaded press either [`y`, `yes`, `1`]. " \
                           + "If no data should be downloaded press either [`n`, `no`, `0`]")
    parser.add_argument('--process', type=str, default='n', 
                        help="If data should be processed press either [`y`, `yes`, `1`]. " \
                           + "If no data should be processed press either [`n`, `no`, `0`]")   
    parser.add_argument('--raw_path', type=str, default='../data/raw/', 
                        help='path of the raw datasets')
    parser.add_argument('--processed_path', type=str, default='../data/processed/', 
                        help='path of the processed datasets')
    
    # Additional optional parameters for processing.
    parser.add_argument('--combined_score_thresh', type=int, default=700,
                        help='threshold below which to cut of gene-gene interactions')
    parser.add_argument('--gdsc', type=str, default='gdsc2',
                        help='filter for GDSC database, options: [`gdsc1`, `gdsc2`, `both`]')
    
    return parser.parse_args()

class HyperParameters:
    def __init__(self, batch_size, lr, train_ratio, val_ratio, num_epochs, seed='12345', num_workers=0):
        self.BATCH_SIZE = batch_size
        self.LR = lr
        self.TRAIN_RATIO = train_ratio
        self.TEST_VAL_RATIO = 1-self.TRAIN_RATIO
        self.VAL_RATIO = val_ratio
        self.NUM_EPOCHS = num_epochs
        self.RANDOM_SEED = seed
        self.NUM_WORKERS = num_workers


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create folder if they don't exist yet.
    Path(args.raw_path).mkdir(parents=True, exist_ok=True)
    Path(args.processed_path).mkdir(parents=True, exist_ok=True)
    
    # File to save logging output to.
    logging.basicConfig(level=logging.DEBUG, filename='arm_1st_v2_logfile', filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")    

    # Initialize processor used for downloading and creation of training datasets.
    processor = Processor(raw_path=args.raw_path, 
                          processed_path=args.processed_path)    
    
    # Download data if necessary.
    if args.download in ['y', 'yes', '1']:
        processor.create_raw_datasets()
        
    # Created training datatsets if necessary.
    if args.process in ['y', 'yes', '1']:
        processor.create_processed_datasets()
        processor.create_gene_gene_interaction_graph()
        processor.create_drug_datasets()
        
    with open(processor.processed_path + 'gdsc2_drm.pkl', 'rb') as f: 
        drm = pickle.load(f)
        print(f"Finished reading drug response matrix: {drm.shape}")

    
    if args.model in ['TabTab', 'tabtab', 'TT', 'tt']:
        # Read cell-line gene matrix.
        with open(processor.processed_path + 'thresh_700_gdsc2_gene_mat.pkl', 'rb') as f: 
            cl_gene_mat = pickle.load(f)
            print("Finished readin cell-line gene matrix:", cl_gene_mat.shape)
            
        # Read drug SMILES fingerprint matrix.
        with open(processor.processed_path + 'gdsc2_smiles_mat.pkl', 'rb') as f:
            smiles_mat = pickle.load(f)
            print(f"Finished reading drug SMILES matrix: {smiles_mat.shape}")                
        
    elif args.model in ['GraphTab', 'graphtab', 'GT', 'gt']:
        # with open(f'{PATH_SUMMARY_DATASETS}{args.model}/{args.version}/drug_response_matrix__gdsc2.pkl', 'rb') as f: 
        #     drm = pickle.load(f)
        #     print(f"Finished reading drug response matrix: {drm.shape}")        
        # Read cell line gene-gene interaction graphs.
        with open(processor.processed_path + 'thresh_700_gdsc2_gene_graphs.pkl', 'rb') as f:
            cl_graphs = pd.read_pickle(f)
            print(f"Finished reading cell-line graphs: {cl_graphs['22RV1']}")
        # Read drug SMILES fingerprint matrix.
        with open(processor.processed_path + 'gdsc2_smiles_dict.pkl', 'rb') as f:
            # drug_name_smiles = pickle.load(f)
            # print(f"Finished reading drug SMILES matrix: {drug_name_smiles.shape}")

            fingerprints_dict = pickle.load(f)
            print(f"Finished reading drug SMILES dict: {len(fingerprints_dict.keys())}")

            # fingerprints_dict = drug_name_smiles.set_index('DRUG_ID').T.to_dict('list')      
    # TODO: add else for other models.
    # TODO: add folder with corresponding datasets for each model type. each folder contains subfolder with dev version
    
    
    elif args.model in ['TabGraph', 'tabgraph', 'TG', 'tg']:
        # Read cell-line gene matrix.
        with open(processor.processed_path + 'thresh_700_gdsc2_gene_mat.pkl', 'rb') as f: 
            cl_gene_mat = pickle.load(f)
            print("Finished readin cell-line gene matrix:", cl_gene_mat.shape)
        # Read drug smiles graphs.
        with open(processor.processed_path + 'gdsc2_smiles_graphs.pkl', 'rb') as f: 
            drug_graphs = pickle.load(f)
            print("Finished reading drug SMILES graphs:", drug_graphs[1003])

    print(f"DRM Number of unique cell-lines: {len(drm.CELL_LINE_NAME.unique())}")


    # --------------- #
    # Train the model #
    # --------------- #
    if args.model in ['TabTab', 'tabgraph', 'TG', 'tg']:
        cl_gene_mat.set_index('CELL_LINE_NAME', inplace=True)
        smiles_mat.set_index('DRUG_ID', inplace=True)

        dataset = TabTabDataset(cl_gene_mat, smiles_mat, drm)
        print("Finished building TabTabDataset!")
        dataset.print_dataset_summary() 

        hyper_params = HyperParameters(batch_size=args.batch_size, 
                                       lr=args.lr, 
                                       train_ratio=args.train_ratio, 
                                       val_ratio=args.val_ratio, 
                                       num_epochs=args.num_epochs, 
                                       seed=args.seed,
                                       num_workers=args.num_workers)

        # Create pytorch geometric DataLoader datasets.
        # TODO: make some args as separate input parameters
        train_loader, test_loader, val_loader = create_tab_tab_datasets(drm, 
                                                                        cl_gene_mat,
                                                                        smiles_mat,
                                                                        hyper_params)
        print("Finished creating pytorch training datasets!")
        print("Number of batches per dataset:")
        print(f"  train : {len(train_loader)}")
        print(f"  test  : {len(test_loader)}")
        print(f"  val   : {len(val_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        model = TabTab_v1().to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr) # TODO: include weight_decay of lr

        # Build the model.
        build_model = BuildTabTabModel(model=model,
                                        criterion=loss_func,
                                        optimizer=optimizer,
                                        num_epochs=args.num_epochs,
                                        train_loader=train_loader,
                                        test_loader=test_loader,
                                        val_loader=val_loader, 
                                        device=device)

        # Train the model.
        print("TRAINING the model")
        performance_stats = build_model.train(build_model.train_loader)        
    
    
    elif args.model == 'GraphTab':
        # Build pytorch dataset.
        graph_tab_dataset = GraphTabDataset(cl_graphs=cl_graphs, drugs=fingerprints_dict, drug_response_matrix=drm)
        print("Finished building GraphTabDataset!")
        graph_tab_dataset.print_dataset_summary() 

        hyper_params = HyperParameters(batch_size=args.batch_size, 
                                      lr=args.lr, 
                                      train_ratio=args.train_ratio, 
                                      val_ratio=args.val_ratio, 
                                      num_epochs=args.num_epochs, 
                                      seed=args.seed,
                                      num_workers=args.num_workers)

        # Create pytorch geometric DataLoader datasets.
        # TODO: make some args as separate input parameters
        train_loader, test_loader, val_loader = create_graph_tab_datasets(drm, cl_graphs, fingerprints_dict, hyper_params)
        print("Finished creating pytorch training datasets!")
        print("Number of batches per dataset:")
        print(f"  train : {len(train_loader)}")
        print(f"  test  : {len(test_loader)}")
        print(f"  val   : {len(val_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        model = GraphTab_v1().to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr) # TODO: include weight_decay of lr

        # Build the model.
        build_model = BuildGraphTabModel(model=model,
                                criterion=loss_func,
                                optimizer=optimizer,
                                num_epochs=args.num_epochs,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                val_loader=val_loader, 
                                device=device)

        # Train the model.
        performance_stats = build_model.train(build_model.train_loader)

        # ONLY USE A SAMPLE
        # sample = drm.sample(1_000)
        # train_set, test_val_set = train_test_split(sample, test_size=0.8, random_state=args.seed)
        # sample_dataset = GraphTabDataset(cl_graphs=cl_graphs, drugs=fingerprints_dict, drug_response_matrix=train_set)
        # print("\ntrain_dataset:")
        # sample_dataset.print_dataset_summary()
        # sample_loader = DataLoader(dataset=sample_dataset, batch_size=2, shuffle=True) 
        # performance_stats = build_model.train(sample_loader)
    elif args.model == 'TabGraph':
        cl_gene_mat.set_index('CELL_LINE_NAME', inplace=True)
        
        dataset = TabGraphDataset(cl_gene_mat, drug_graphs, drm)
        print("Finished building TabGraphDataset!")
        dataset.print_dataset_summary() 

        hyper_params = HyperParameters(batch_size=args.batch_size, 
                                      lr=args.lr, 
                                      train_ratio=args.train_ratio, 
                                      val_ratio=args.val_ratio, 
                                      num_epochs=args.num_epochs, 
                                      seed=args.seed,
                                      num_workers=args.num_workers)

        # Create pytorch geometric DataLoader datasets.
        # TODO: make some args as separate input parameters
        train_loader, test_loader, val_loader = create_tab_graph_datasets(drm, cl_gene_mat, drug_graphs, hyper_params)
        print("Finished creating pytorch training datasets!")
        print("Number of batches per dataset:")
        print(f"  train : {len(train_loader)}")
        print(f"  test  : {len(test_loader)}")
        print(f"  val   : {len(val_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        model = TabGraph_v1().to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr) # TODO: include weight_decay of lr

        # Build the model.
        build_model = BuildTabGraphModel(model=model,
                                        criterion=loss_func,
                                        optimizer=optimizer,
                                        num_epochs=args.num_epochs,
                                        train_loader=train_loader,
                                        test_loader=test_loader,
                                        val_loader=val_loader, 
                                        device=device)

        # Train the model.
        performance_stats = build_model.train(build_model.train_loader)        
    # elif args.model == 'TabTab':
    #     dataset = TabTabDataset()


    torch.save({
        'epoch': args.num_epochs, # TODO: add here current epoch. For that the epochs must run in the main.
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'model_state_dict': build_model.model.state_dict(),
        'optimizer_state_dict': build_model.optimizer.state_dict(),
        'train_performances': performance_stats['train'],
        'val_performances': performance_stats['val']
    }, PERFORMANCES + f'model_performance_{args.model}.pth')
        


if __name__ == "__main__":
    main()        
