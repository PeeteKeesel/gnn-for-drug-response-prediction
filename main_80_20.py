import random
import logging
import pickle
import copy
import torch
import torch.nn as nn
import numpy    as np
import pandas   as pd

from argparse                      import ArgumentParser
from pathlib                       import Path
from src.models.TabTab.tab_tab_early_stopping         import TabTabDataset, create_tt_loaders, BuildTabTabModel, TabTab
from src.models.GraphTab.graph_tab_early_stopping     import GraphTabDataset, create_gt_loaders, BuildGraphTabModel, GraphTab
from src.models.GraphGraph.graph_graph_early_stopping import GraphGraphDataset, create_gg_loaders, BuildGraphGraphModel, GraphGraph

# from src.models.TabTab.tab_tab     import TabTabDataset, create_tab_tab_datasets, BuildTabTabModel, TabTab_v1
# from src.models.TabTab.tab_tab_early_stopping   import TabTabDataset, create_tab_tab_datasets, BuildTabTabModel, TabTab
# from src.models.GraphTab.graph_tab import GraphTabDataset, create_graph_tab_datasets, BuildGraphTabModel, GraphTab_v1, GraphTab_v2
# from src.models.GraphTab.graph_tab_early_stopping import GraphTabDataset, create_graph_tab_datasets, BuildGraphTabModel, GraphTab_v1, GraphTab_v2
# from src.models.TabGraph.tab_graph import TabGraphDataset, create_tab_graph_datasets, BuildTabGraphModel, TabGraph_v1
from src.preprocess.processor      import Processor

from ignite.engine           import Engine, Events, create_supervised_trainer
from ignite.handlers         import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

PERFORMANCES = 'performances/'


def parse_args():
    parser = ArgumentParser(description='GNNs for Drug Response Prediction in Cancer')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=1_000, 
                        help='the batch size (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='weight decay (default: 0.0001)')    
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.5, 
                        help='validation set ratio inside the test set (default: 0.5)')
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                        help='testing set ratio (default: 0.2)')    
    parser.add_argument('--num_epochs', type=int, default=5, 
                        help='number of epochs (default: )')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='number of workers for DataLoader (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='dropout probability (default: 0.1)')
    parser.add_argument('--kfolds', type=int, default=5, 
                        help='number of folds for cross validation (default: 5)')     
    parser.add_argument('--model', type=str, default='GraphTab', 
                        help='name of the model to run, options: ' + \
                             '[`TabTab`, `GraphTab`, `TabGraph`, `GraphGraph`,' + \
                             ' `tabtab`, `graphtab`, `tabgraph`, `graphgraph`, ' + \
                             ' `TT`, `GT`, `TG`, `GG`, `tt`, `gt`, `tg`, `gg` ]')
    parser.add_argument('--conv_type', type=str, default='',
                        help='convolution type for the GNNs, options [`GCNConv`, `GATConv`]')
    parser.add_argument('--conv_layers', type=int, default=2,
                        help='number of convolutional layers for the GNNs, options [2]')
    parser.add_argument('--global_pooling', type=str, default='max',
                        help='type of global pooling (default: max)')
    parser.add_argument('--version', type=str, default='v1', 
                        help='model version to run')
    parser.add_argument('--download', type=str, default='n', 
                        help="If raw data should be downloaded press either [`y`, `yes`, `1`]. " \
                           + "If no data should be downloaded press either [`n`, `no`, `0`]")
    parser.add_argument('--process', type=str, default='n', 
                        help="If data should be processed press either [`y`, `yes`, `1`]. " \
                           + "If no data should be processed press either [`n`, `no`, `0`]")   
    parser.add_argument('--build_model', type=str, default='y', 
                        help="If model should be build")       
    parser.add_argument('--raw_path', type=str, default='../data/raw/', 
                        help='path of the raw datasets')
    parser.add_argument('--processed_path', type=str, default='../data/processed/', 
                        help='path of the processed datasets')
    parser.add_argument('--logging_path', type=str, default='',
                        help='path for the logging results')
    parser.add_argument('--early_stopping_threshold', type=float, default=20,
                        help='early stopping threshold')
    parser.add_argument('--ablation_subset', type=str, default='',
                        help='node features for ablation studies in [gexpr, cnvg, cnvp, mut]')    
    parser.add_argument('--nr_node_features', type=int, default=4,
                        help='number of node features')
    
    # Additional optional parameters for processing.
    parser.add_argument('--combined_score_thresh', type=int, default=990,
                        help='threshold below which to cut of gene-gene interactions')
    parser.add_argument('--gdsc', type=str, default='gdsc2',
                        help='filter for GDSC database, options: [`gdsc1`, `gdsc2`, `both`]')
    parser.add_argument('--file_ending', type=str, default='',
                        help='ending of final models file name')
    
    return parser.parse_args()
     

def log_args_summary(args):
    logging.info("ARGUMENTS SUMMARY")
    logging.info("=================")
    logging.info(f"seed                     : {args.seed}")
    logging.info(f"batch_size               : {args.batch_size}")
    logging.info(f"lr                       : {args.lr}")
    logging.info(f"num_epochs               : {args.num_epochs}") 
    logging.info(f"num_workers              : {args.num_workers}")
    logging.info(f"dropout                  : {args.dropout}")
    logging.info(f"kfolds                   : {args.kfolds}")
    logging.info(f"conv_type                : {args.conv_type}")
    logging.info(f"conv_layers              : {args.conv_layers}")
    logging.info(f"global_pooling           : global_{args.global_pooling}_pooling")    
    logging.info(f"early_stopping_threshold : {args.early_stopping_threshold}")    
    logging.info(f"combined_score_thresh    : {args.combined_score_thresh}")
    logging.info(f"gdsc                     : {args.gdsc}")
    logging.info(f"file_ending              : {args.file_ending}")
    

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create folder for datasets and performance results if they don't exist yet.
    Path(args.raw_path).mkdir(parents=True, exist_ok=True)
    Path(args.processed_path).mkdir(parents=True, exist_ok=True)
    Path(args.processed_path + args.gdsc + '/' + str(args.combined_score_thresh)).mkdir(parents=True, exist_ok=True)
    
    # File to save logging output to.
    logging.basicConfig(
        level=logging.INFO, filemode="a+",
        filename=PERFORMANCES + \
            args.logging_path + \
            f'logfile_model_{args.model.lower()}_{args.version}_{args.gdsc}_{args.combined_score_thresh}_{args.seed}_{args.file_ending}',
        format="%(asctime)-15s %(levelname)-8s %(message)s"
    )  

    # Initialize processor used for downloading and creation of training datasets.
    processor = Processor(
        raw_path=args.raw_path, 
        processed_path=args.processed_path,
        combined_score_thresh=args.combined_score_thresh,
        gdsc=args.gdsc
    )
    
    # Download data if necessary.
    if args.download in ['y', 'yes', '1']:
        processor.create_raw_datasets()
        
    # Created training datatsets if necessary.
    if args.process in ['y', 'yes', '1']:
        processor.create_processed_datasets()
        processor.create_gene_gene_interaction_graph()
        processor.create_drug_datasets()
        
    if not args.build_model:
        return

    # -----------------------------------     
    # --- Drug response matrix import ---
    # -----------------------------------
    with open(processor.processed_path + 'gdsc2_drm.pkl', 'rb') as f: 
        drm = pickle.load(f)
        logging.info(f"Finished reading drug response matrix: {drm.shape}")
        
    logging.info(f"DRM Number of unique cell-lines: {len(drm.CELL_LINE_NAME.unique())}")        
    # ------------------------------
    # --- TabTab dataset imports ---
    # ------------------------------    
    if args.model in ['TabTab', 'tabtab', 'TT', 'tt']:
        # Read cell-line gene matrix.
        with open(processor.gdsc_thresh_path + \
                  f'thresh_{processor.gdsc.lower()}_{processor.combined_score_thresh}_gene_mat.pkl', 'rb') as f: 
            cl_gene_mat = pickle.load(f)
            logging.info(f"Finished reading cell-line gene matrix: {cl_gene_mat.shape}")
            
        # Read drug SMILES fingerprint matrix.
        with open(processor.gdsc_path + \
                  f'{processor.gdsc.lower()}_smiles_mat.pkl', 'rb') as f:
            smiles_mat = pickle.load(f)
            logging.info(f"Finished reading drug SMILES matrix: {smiles_mat.shape}")
    # --------------------------------
    # --- GraphTab dataset imports ---
    # --------------------------------    
    elif args.model in ['GraphTab', 'graphtab', 'GT', 'gt']:       
        # Read cell line gene-gene interaction graphs.
        with open(processor.gdsc_thresh_path + \
                  f'thresh_{processor.gdsc.lower()}_{processor.combined_score_thresh}_gene_graphs.pkl', 'rb') as f:
            cl_graphs = pd.read_pickle(f)
            logging.info(f"Finished reading cell-line graphs: {cl_graphs['22RV1']}")
        # Read drug SMILES fingerprint matrix.
        with open(processor.gdsc_path + \
                  f'{processor.gdsc.lower()}_smiles_dict.pkl', 'rb') as f:
            fingerprints_dict = pickle.load(f)
            logging.info(f"Finished reading drug SMILES dict: {len(fingerprints_dict.keys())}")
    # --------------------------------
    # --- TabGraph dataset imports ---
    # --------------------------------    
    elif args.model in ['TabGraph', 'tabgraph', 'TG', 'tg']:
        # Read cell-line gene matrix.
        with open(processor.gdsc_thresh_path + \
                  f'thresh_{processor.gdsc.lower()}_{processor.combined_score_thresh}_gene_mat.pkl', 'rb') as f: 
            cl_gene_mat = pickle.load(f)
            logging.info(f"Finished reading cell-line gene matrix: {cl_gene_mat.shape}")
        # Read drug smiles graphs.
        with open(processor.gdsc_path + \
                  f'{processor.gdsc.lower()}_smiles_graphs.pkl', 'rb') as f: 
            drug_graphs = pickle.load(f)
            logging.info(f"Finished reading drug SMILES graphs: {drug_graphs[1003]}")
    # ----------------------------------
    # --- GraphGraph dataset imports ---
    # ----------------------------------
    elif args.model in ['GraphGraph', 'graphgraph', 'GG', 'gg']:
        # Read cell line gene-gene interaction graphs.
        with open(processor.gdsc_thresh_path + \
                  f'thresh_{processor.gdsc.lower()}_{processor.combined_score_thresh}_gene_graphs.pkl', 'rb') as f:
            cl_graphs = pd.read_pickle(f)
            logging.info(f"Finished reading cell-line graphs: {cl_graphs['22RV1']}")
        # Read drug smiles graphs.
        with open(processor.gdsc_path + \
                  f'{processor.gdsc.lower()}_smiles_graphs.pkl', 'rb') as f: 
            drug_graphs = pickle.load(f)
            logging.info(f"Finished reading drug SMILES graphs: {drug_graphs[1003]}")

    # Log arguments summary.
    log_args_summary(args)
            
    # --------------- #
    # Train the model #
    # --------------- #
    #
    #
    # -----------------------------
    # --- TabTab model training ---
    # -----------------------------    
    if args.model in ['TabTab', 'tabgraph', 'TG', 'tg']:
        cl_gene_mat.set_index('CELL_LINE_NAME', inplace=True)
        smiles_mat.set_index('DRUG_ID', inplace=True)

        tab_tab_dataset = TabTabDataset(
            cl_gene_mat, 
            smiles_mat, 
            drm
        )
        logging.info("Finished building TabTabDataset!")
        tab_tab_dataset.print_dataset_summary()   

        drm_train, drm_test = train_test_split(
            drm, 
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=drm['CELL_LINE_NAME']
        )
            
        # Create data loaders.
        train_loader, test_loader = create_tt_loaders(
            drm_train,
            drm_test,
            cl_gene_mat,
            smiles_mat,
            args
        )
        logging.info(f"{4*' '}Finished creating pytorch training datasets!")
        logging.info(f"{4*' '}Number of batches per dataset:")
        logging.info(f"{8*' '}train : {len(train_loader)}")      
        logging.info(f"{8*' '}test  : {len(test_loader)}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        
        # Initialize model.
        model = TabTab(
            cell_inp_dim=cl_gene_mat.shape[1],
            dropout=args.dropout
        )
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logging.info(f"GPU Usage: {torch.cuda.max_memory_allocated(device=device)}")
        
        # Enable multi-GPU parallelization if feasible.
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).to(device)
        else:
            model =  model.to(device)
        
        # Define loss function and optimizer.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Build the model.
        build_model = BuildTabTabModel(
            model=model, 
            criterion=loss_func, 
            optimizer=optimizer,
            num_epochs=args.num_epochs, 
            train_loader=train_loader,
            test_loader=test_loader,
            early_stopping_threshold=args.early_stopping_threshold,            
            device=device
        )
        logging.info(build_model.model)
        
        # Train the model on the training fold and evaluate on the test fold.
        logging.info("TRAINING the model")
        performance_stats = build_model.train(
            build_model.train_loader
        )    
    # 
    #
    # -------------------------------
    # --- GraphTab model training ---
    # -------------------------------    
    elif args.model in ['GraphTab', 'graphtab', 'GT', 'gt']:
        
        graph_tab_dataset = GraphTabDataset(
            cl_graphs, 
            fingerprints_dict, 
            drm
        )
        logging.info("Finished building GraphTabDataset!")
        graph_tab_dataset.print_dataset_summary()
        
        drm_train, drm_test = train_test_split(
            drm, 
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=drm['CELL_LINE_NAME']
        )
        
        # Create data loaders.
        train_loader, test_loader = create_gt_loaders(
            drm_train,
            drm_test,
            cl_graphs,
            fingerprints_dict,
            args
        )
        
        logging.info(f"{4*' '}Finished creating pytorch training datasets!")
        logging.info(f"{4*' '}Number of batches per dataset:")
        logging.info(f"{8*' '}train : {len(train_loader)}")      
        logging.info(f"{8*' '}test  : {len(test_loader)}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        
        # Initialize model.
        model = GraphTab(
            dropout=args.dropout,
            conv_type=args.conv_type,
            conv_layers=args.conv_layers,
            global_pooling=args.global_pooling
        ) 
        
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logging.info(f"GPU Usage: {torch.cuda.max_memory_allocated(device=device)}") 
        
        # Enable multi-GPU parallelization if feasible.
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).to(device)
        else:
            model =  model.to(device)            
        
        # Define loss function and optimizer.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )            
        
        # Build the model.
        build_model = BuildGraphTabModel(
            model=model, 
            criterion=loss_func, 
            optimizer=optimizer,
            num_epochs=args.num_epochs, 
            train_loader=train_loader,
            test_loader=test_loader,
            early_stopping_threshold=args.early_stopping_threshold,            
            device=device
        )
        logging.info(build_model.model)
        
        # Train the model on the training fold and evaluate on the test fold.
        logging.info("TRAINING the model")
        performance_stats = build_model.train(
            build_model.train_loader
        )           
    #
    #
    # -------------------------------
    # --- TabGraph model training ---
    # -------------------------------    
    elif args.model == 'TabGraph':
        cl_gene_mat.set_index('CELL_LINE_NAME', inplace=True)
        
        dataset = TabGraphDataset(cl_gene_mat, drug_graphs, drm)
        logging.info("Finished building TabGraphDataset!")
        dataset.print_dataset_summary() 

        hyper_params = HyperParameters(
            batch_size=args.batch_size, 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            num_epochs=args.num_epochs, 
            seed=args.seed,
            num_workers=args.num_workers
        )
        logging.info(hyper_params())        

        # Create pytorch geometric DataLoader datasets.
        # TODO: make some args as separate input parameters
        train_loader, test_loader, val_loader = create_tab_graph_datasets(
            drm, 
            cl_gene_mat, 
            drug_graphs, 
            hyper_params
        )
        logging.info("Finished creating pytorch training datasets!")
        logging.info("Number of batches per dataset:")
        logging.info(f"  train : {len(train_loader)}")
        logging.info(f"  test  : {len(test_loader)}")
        logging.info(f"  val   : {len(val_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")

        model = TabGraph_v1(cl_gene_mat.shape[1]).to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), 
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        # Build the model.
        build_model = BuildTabGraphModel(
            model=model,
            criterion=loss_func,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader, 
            device=device
        )

        # Train the model.
        performance_stats = build_model.train(build_model.train_loader)        
    #
    #
    # ---------------------------------
    # --- GraphGraph model training ---
    # ---------------------------------   
    elif args.model in ['GraphGraph', 'graphgraph', 'GG', 'gg']:
        
        # --- Perform ablation study if specified ---
        if args.ablation_subset:
            node_feature_mapping = {
                'gexpr': 0,
                'cnvg': 1,
                'cnvp': 2,
                'mut': 3
            }
            
            logging.info(f"\n\nPerforming ablation study only for: {args.ablation_subset}!")
            logging.info("============================================================")
            idx = node_feature_mapping.get(args.ablation_subset)
            cl_graphs_subset = {}

            for cl, G in cl_graphs.items():
                G_temp = copy.deepcopy(G)
                G_temp.x = G_temp.x[:, idx].unsqueeze(dim=-1)
                cl_graphs_subset[cl] = G_temp
                
            cl_graphs = cl_graphs_subset
        
        graph_graph_dataset = GraphGraphDataset(
            cl_graphs, 
            drug_graphs, 
            drm
        )
        logging.info("Finished building GraphGraphDataset!")
        graph_graph_dataset.print_dataset_summary()
        
        drm_train, drm_test = train_test_split(
            drm, 
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=drm['CELL_LINE_NAME']
        )
        
        # Create data loaders.
        train_loader, test_loader = create_gg_loaders(
            drm_train,
            drm_test,
            cl_graphs,
            drug_graphs,
            args
        )
        
        logging.info(f"{4*' '}Finished creating pytorch training datasets!")
        logging.info(f"{4*' '}Number of batches per dataset:")
        logging.info(f"{8*' '}train : {len(train_loader)}")      
        logging.info(f"{8*' '}test  : {len(test_loader)}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        
        # Initialize model.
        model = GraphGraph(
            dropout=args.dropout,
            conv_type=args.conv_type,
            conv_layers=args.conv_layers,
            global_pooling=args.global_pooling,
            nr_node_features=args.nr_node_features
        ) 
        
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logging.info(f"GPU Usage: {torch.cuda.max_memory_allocated(device=device)}") 
        
        # Enable multi-GPU parallelization if feasible.
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).to(device)
        else:
            model =  model.to(device)            
        
        # Define loss function and optimizer.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )            
        
        # Build the model.
        build_model = BuildGraphGraphModel(
            model=model, 
            criterion=loss_func, 
            optimizer=optimizer,
            num_epochs=args.num_epochs, 
            train_loader=train_loader,
            test_loader=test_loader,
            early_stopping_threshold=args.early_stopping_threshold,            
            device=device
        )
        logging.info(build_model.model)
        
        # Train the model on the training fold and evaluate on the test fold.
        logging.info("TRAINING the model")
        performance_stats = build_model.train(
            build_model.train_loader
        )
        
        
    torch.save({
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'train_ratio': args.train_ratio,
        'test_ratio': args.test_ratio,
        'early_stopping_thresh': args.early_stopping_threshold,
        'model_state_dict': build_model.model.state_dict(),
        'optimizer_state_dict': build_model.optimizer.state_dict(),
        'performances': performance_stats
    }, PERFORMANCES + args.logging_path + f'model_performance_{args.model}_{args.version}_{args.gdsc.lower()}_{args.combined_score_thresh}_{args.seed}_{args.conv_type}_{args.file_ending}.pth')
        


if __name__ == "__main__":
    main()