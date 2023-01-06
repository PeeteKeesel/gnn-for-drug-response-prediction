import random
import logging
import pickle
import torch
import torch.nn as nn
import numpy    as np
import pandas   as pd

from sklearn.model_selection       import train_test_split
from argparse                      import ArgumentParser
from pathlib                       import Path
from src.models.TabTab.tab_tab     import TabTabDataset, create_tab_tab_datasets, BuildTabTabModel, TabTab_v1
from src.models.GraphTab.graph_tab import GraphTabDataset, create_graph_tab_datasets, create_gt_loaders, BuildGraphTabModel, GraphTab_v1, GraphTab_v2
from src.models.TabGraph.tab_graph import TabGraphDataset, create_tab_graph_datasets, BuildTabGraphModel, TabGraph_v1
from src.preprocess.processor      import Processor
from skopt                         import gp_minimize
from skopt.space                   import Real, Integer
from sklearn.model_selection       import KFold
from skopt                         import gbrt_minimize
from functools                     import partial


PERFORMANCES = 'performances/'


def parse_args():
    parser = ArgumentParser(description='GNNs for Drug Response Prediction in Cancer')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=1_000, 
                        help='the batch size (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.5, 
                        help='validation set ratio inside the test set (default: 0.5)')
    parser.add_argument('--num_epochs', type=int, default=5, 
                        help='number of epochs (default: )')
    parser.add_argument('--num_workers', type=int, default=24, 
                        help='number of workers for DataLoader (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='dropout probability (default: 0.1)')
    parser.add_argument('--kfolds', type=float, default=5, 
                        help='number of folds for cross validation (default: 5)')    
    parser.add_argument('--model', type=str, default='GraphTab', 
                        help='name of the model to run, options: ' + \
                             '[`TabTab`, `GraphTab`, `TabGraph`, `GraphGraph`,' + \
                             ' `tabtab`, `graphtab`, `tabgraph`, `graphgraph`, ' + \
                             ' `TT`, `GT`, `TG`, `GG`, `tt`, `gt`, `tg`, `gg` ]')
    parser.add_argument('--version', type=str, default='v1', 
                        help='model version to run')
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
    parser.add_argument('--combined_score_thresh', type=int, default=990,
                        help='threshold below which to cut of gene-gene interactions')
    parser.add_argument('--gdsc', type=str, default='gdsc2',
                        help='filter for GDSC database, options: [`gdsc1`, `gdsc2`, `both`]')
    parser.add_argument('--file_ending', type=str, default='',
                        help='ending of final models file name')
    
    return parser.parse_args()

class HyperParameters:
    def __init__(self, batch_size, lr, train_ratio, val_ratio, num_epochs, seed='42', num_workers=0):
        self.BATCH_SIZE = batch_size
        self.LR = lr
        self.TRAIN_RATIO = train_ratio
        self.TEST_VAL_RATIO = 1-self.TRAIN_RATIO
        self.VAL_RATIO = val_ratio
        self.NUM_EPOCHS = num_epochs
        self.RANDOM_SEED = seed
        self.NUM_WORKERS = num_workers
        
    def __call__(self):
        logging.info("HyperParameters")
        logging.info("===============")
        logging.info(f"batch_size: {self.BATCH_SIZE}")
        logging.info(f"learning_rate: {self.LR}")
        logging.info(f"train_ratio: {self.TRAIN_RATIO}")
        logging.info(f"val_ratio: {self.VAL_RATIO}")
        logging.info(f"test_ratio: {1-self.VAL_RATIO}")
        logging.info(f"num_epochs: {self.NUM_EPOCHS}") 
        logging.info(f"num_workers: {self.NUM_WORKERS}")
        logging.info(f"random_seed: {self.RANDOM_SEED}")     
        

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
        level=logging.DEBUG, filemode="a+",
        filename=PERFORMANCES + \
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
        
    # -------------------------------------------------------------------------
    # --- Drug response matrix ---
    with open(processor.processed_path + 'gdsc2_drm.pkl', 'rb') as f: 
        drm = pickle.load(f)
        logging.info(f"Finished reading drug response matrix: {drm.shape}")
        
    logging.info(f"DRM Number of unique cell-lines: {len(drm.CELL_LINE_NAME.unique())}")        

    # --- TabTab dataset imports ---
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
    # --- GraphTab dataset imports ---
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
    
    # --- TabGraph dataset imports ---    
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

    # -------------------------------------------------------------------------
    
    # --------------- #
    # Train the model #
    # --------------- #
    # --- TabTab model training ---
    if args.model in ['TabTab', 'tabgraph', 'TG', 'tg']:
        cl_gene_mat.set_index('CELL_LINE_NAME', inplace=True)
        smiles_mat.set_index('DRUG_ID', inplace=True)

        dataset = TabTabDataset(cl_gene_mat, smiles_mat, drm)
        logging.info("Finished building TabTabDataset!")
        dataset.print_dataset_summary() 

        hyper_params = HyperParameters(
            batch_size=args.batch_size, 
            lr=args.lr, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            num_epochs=args.num_epochs, 
            seed=args.seed,
            num_workers=args.num_workers
        )
        logging.info(hyper_params())        

        # Create pytorch geometric DataLoader datasets.
        # TODO: make some args as separate input parameters
        train_loader, test_loader, val_loader = create_tab_tab_datasets(
            drm, 
            cl_gene_mat,
            smiles_mat,
            hyper_params
        )
        logging.info("Finished creating pytorch training datasets!")
        logging.info("Number of batches per dataset:")
        logging.info(f"  train : {len(train_loader)}")
        logging.info(f"  test  : {len(test_loader)}")
        logging.info(f"  val   : {len(val_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")

        model = TabTab_v1(cl_gene_mat.shape[1]).to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr) # TODO: include weight_decay of lr

        # Build the model.
        build_model = BuildTabTabModel(
            model=model, 
            criterion=loss_func, 
            optimizer=optimizer,
            num_epochs=args.num_epochs, 
            train_loader=train_loader,
            test_loader=test_loader, 
            val_loader=val_loader, 
            device=device
        )
        logging.info(build_model.model) 

        # Train the model.
        logging.info("TRAINING the model")
        performance_stats = build_model.train(build_model.train_loader)        
   
    # --- GraphTab model training ---    
    elif args.model in ['GraphTab', 'graphtab', 'GT', 'gt']:
        # Build pytorch dataset.
        graph_tab_dataset = GraphTabDataset(
            cl_graphs=cl_graphs, 
            drugs=fingerprints_dict, 
            drug_response_matrix=drm
        )
        logging.info("Finished building GraphTabDataset!")
        graph_tab_dataset.print_dataset_summary() 

        hyper_params = HyperParameters(
            batch_size=args.batch_size, 
            lr=args.lr, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            num_epochs=args.num_epochs, 
            seed=args.seed,
            num_workers=args.num_workers
        )
        logging.info(hyper_params())
        
        train_val_set, test_set = train_test_split(
            drm, 
            test_size=0.1,
            random_state=args.seed,
            stratify=drm['CELL_LINE_NAME']
        )
        
        # Define the search spaces.
        learning_rate_space = Real(.001, .1, name='learning_rate')
        batch_size_space = Integer(10, 1_000, name='batch_size')
        weight_decay_space = Real(0, .001, name='weight_decay')
        
        def optimize_gt(params):
            batch_size = params[0]    
            weight_decay = params[1]
            learning_rate = params[2]
            
            # Use 5-fold cross-validation to evaluate the model's performance
            kf, performances = KFold(n_splits=args.kfolds), []
            for i, (i_train, i_val) in enumerate(kf.split(train_val_set)):
                logging.info(f"KFold iteration {i}")  
                train_set = train_val_set.iloc[i_train]
                val_set = train_val_set.iloc[i_val]                
                
                train_loader, val_loader, test_loader = create_gt_loaders(
                    train_set, 
                    val_set,
                    test_set,
                    cl_graphs, 
                    fingerprints_dict, 
                    args,
                    batch_size
                )
                logging.info("Finished creating pytorch training datasets!")
                logging.info("Number of batches per dataset:")
                logging.info(f"  train : {len(train_loader)}")
                logging.info(f"  val   : {len(val_loader)}")        
                logging.info(f"  test  : {len(test_loader)}")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logging.info(f"device: {device}")   
                
                match args.version:
                    case 'v1': 
                        model = GraphTab_v1().to(device)
                    case 'v2':
                        model = GraphTab_v2().to(device)
                    case _:
                        raise NotImplementedError(f"Given model version {args.version} is not implemented for GraphTab!")
                                  
                loss_func = nn.MSELoss()
                optimizer = torch.optim.Adam(params=model.parameters(), 
                                             lr=learning_rate,
                                             weight_decay=weight_decay)

                # Build the model.
                build_model = BuildGraphTabModel(
                    model=model,
                    criterion=loss_func,
                    optimizer=optimizer,
                    num_epochs=args.num_epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,            
                    test_loader=test_loader, 
                    device=device
                )
                logging.info(build_model.model)      

                # Train the model on the training fold.
                performance_stats = build_model.train(build_model.train_loader)

                # Evaluate the model on the validation fold.
                mse_va, rmse_va, mae_va, r2_va, pcc_va, scc_va, _, _ = build_model.validate(build_model.val_loader)        

                performances.append(rmse_va)

            # Return the negative of the average performance (since skopt minimizes the objective function)
            return -np.mean(performances)           
        
        # Use the gp_minimize function to optimize the objective function.
        opt_params = gp_minimize(
            optimize_gt, 
            (
                batch_size_space,
                weight_decay_space,
                learning_rate_space
#                 'kfold_splits': args.kfolds,
#                 'train_val_set': train_val_set,
#                 'test_set': test_set,
#                 'args': args
            ),
            n_calls=50
        )
        
        # Extract the optimal values of the hyperparameters.
        optimal_batch_size = opt_params.x[0]
        optimal_weight_decay = opt_params.x[1]
        optimal_learning_rate = opt_params.x[2]

        # Create pytorch geometric DataLoader datasets.
        # TODO: make some args as separate input parameters
        train_loader, val_loader, test_loader = create_gt_loaders(
            train_val_set, 
            cl_graphs, 
            fingerprints_dict, 
            params['args'],
            optimal_batch_size,
            params['test_set']
        )
        logging.info("Finished creating pytorch training datasets!")
        logging.info("Number of batches per dataset:")
        logging.info(f"  train : {len(train_loader)}")
        logging.info(f"  val   : {len(val_loader)}")        
        logging.info(f"  test  : {len(test_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        
        match args.version:
            case 'v1': 
                model = GraphTab_v1().to(device)
            case 'v2':
                model = GraphTab_v2().to(device)
            case _:
                raise NotImplementedError(f"Given model version {args.version} is not implemented for GraphTab!")
            
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), 
                                     lr=optimal_learning_rate,
                                     weight_decay=optimal_weight_decay)
        # TODO: include weight_decay of lr
        # check https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer.py#L32

        # Build the model.
        build_model = BuildGraphTabModel(
            model=model,
            criterion=loss_func,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,             
            test_loader=test_loader,
            device=device
        )
        logging.info(build_model.model)      

        # Train the model.
        performance_stats = build_model.train(build_model.train_loader)
        
        
        # ONLY USE A SAMPLE
        # sample = drm.sample(1_000)
        # train_set, test_val_set = train_test_split(sample, test_size=0.8, random_state=args.seed)
        # sample_dataset = GraphTabDataset(cl_graphs=cl_graphs, drugs=fingerprints_dict, drug_response_matrix=train_set)
        # logging.info("\ntrain_dataset:")
        # sample_dataset.print_dataset_summary()
        # sample_loader = DataLoader(dataset=sample_dataset, batch_size=2, shuffle=True) 
        # performance_stats = build_model.train(sample_loader)

    # --- TabGraph model training ---    
    elif args.model == 'TabGraph':
        cl_gene_mat.set_index('CELL_LINE_NAME', inplace=True)
        
        dataset = TabGraphDataset(cl_gene_mat, drug_graphs, drm)
        logging.info("Finished building TabGraphDataset!")
        dataset.print_dataset_summary() 

        hyper_params = HyperParameters(
            batch_size=args.batch_size, 
            lr=args.lr, 
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
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr) # TODO: include weight_decay of lr

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

    torch.save({
        'epoch': args.num_epochs, # TODO: maybe add here current epoch. For that the epochs must run in the main.
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'model_state_dict': build_model.model.state_dict(),
        'optimizer_state_dict': build_model.optimizer.state_dict(),
        'train_performances': performance_stats['train'],
        'val_performances': performance_stats['val'],
        'test_performance': performance_stats['test']
    }, PERFORMANCES + f'Bayes_model_performance_{args.model}_{args.version}_{args.gdsc.lower()}_{args.combined_score_thresh}_{args.seed}_{args.file_ending}.pth')
        


if __name__ == "__main__":
    main()        
