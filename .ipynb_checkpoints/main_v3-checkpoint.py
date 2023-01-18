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
# from src.models.GraphTab.graph_tab import GraphTabDataset, create_graph_tab_datasets, create_gt_loaders, BuildGraphTabModel, GraphTab_v1, GraphTab_v2, create_gt_final_datasets, create_gt_loaders
from src.models.GraphTab.graph_tab_v2 import GraphTabDataset, create_graph_tab_datasets, create_gt_loaders, BuildGraphTabModel, GraphTab_v1, GraphTab_v2, create_gt_final_datasets, create_gt_loaders
from src.models.TabGraph.tab_graph import TabGraphDataset, create_tab_graph_datasets, BuildTabGraphModel, TabGraph_v1
from src.preprocess.processor      import Processor
from skopt                         import gbrt_minimize
from skopt.space                   import Real, Integer, Categorical
from sklearn.model_selection       import KFold
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
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                        help='validation set ratio inside the test set (default: 0.5)')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                        help='test set ratio (default: 0.1)')    
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
        
def print_args_summary(args):
    logging.info("Args Summary")
    logging.info("============")
    logging.info(f"{'batch_size':>20}: {args.batch_size}")
    logging.info(f"{'learning_rate':>20}: {args.lr}")
    logging.info(f"{'train_ratio':>20}: {args.train_ratio}")
    logging.info(f"{'val_ratio':>20}: {args.val_ratio}")
    logging.info(f"{'test_ratio':>20}: {args.test_ratio}")
    logging.info(f"{'num_epochs':>20}: {args.num_epochs}") 
    logging.info(f"{'num_workers':>20}: {args.num_workers}")
    logging.info(f"{'random_seed':>20}: {args.seed}")      
        
        
def objective_gt(params, args, device, datasets):
    """
    Defines the objective function for GraphTab.
    """
    learning_rate, weight_decay, batch_size = params
    train_val_set, test_set, cl_graphs, fingerprints_dict = datasets
    logging.info(f"{4*' '}learning_rate : {learning_rate}")
    logging.info(f"{4*' '}weight_decay  : {weight_decay}")
    logging.info(f"{4*' '}batch_size    : {batch_size}")
    print(type(batch_size))
    
    args.batch_size = batch_size

    # Initialize model and model class.
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

    # Compute cross-validation score.
    kfold = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
#     cv_performances = {}
    cv_mse_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []
    cv_pcc_scores = []
    cv_scc_scores = []
    logging.info(f"Running {args.kfolds}-Fold CV")
    logging.info("===================")
    logging.info(f"{4*' '}Parameter Set")
    logging.info(f"{4*' '}-------------")
    logging.info(f"{8*' '}learning_rate: {learning_rate}")
    logging.info(f"{8*' '}weight_decay : {weight_decay}")
    logging.info(f"{8*' '}batch_size   : {batch_size}")    
    for i, (train_i, val_i) in enumerate(kfold.split(train_val_set), 1):
        logging.info(f"{12*' '}Fold iteration k={i:2.0f}...")
        
        train_set = train_val_set.iloc[train_i]
        val_set = train_val_set.iloc[val_i]

        # Create loaders.
        train_loader, val_loader, test_loader = create_gt_loaders(
            [train_set, val_set, test_set],
            cl_graphs, 
            fingerprints_dict,
            args
        )
        
        gt_cls = BuildGraphTabModel(
            model=model,
            criterion=loss_func,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader, 
            device=device
        )
        logging.info(gt_cls.model)        

        # Set the loader attributes such that they can be used inside the class.
        gt_cls.train_loader = train_loader
        gt_cls.val_ratio = val_loader
        gt_cls.test_loader = test_loader

        # Train the model on the training fold.
        performances = gt_cls.train(gt_cls.train_loader)

#         cv_performances[f'k{i}_val'] = performances.get('val')
        cv_mse_scores.append(performances.get('val').get('mse'))
        cv_rmse_scores.append(performances.get('val').get('rmse'))
        cv_mae_scores.append(performances.get('val').get('mae'))
        cv_r2_scores.append(performances.get('val').get('r2'))
        cv_pcc_scores.append(performances.get('val').get('pcc'))
        cv_scc_scores.append(performances.get('val').get('scc'))       
        
    scores_as_tensor = torch.tensor(cv_mse_scores).detach().cpu().numpy()
    logging.info(f"{args.kfolds}-Fold CV Results")
    logging.info("=================")
    logging.info(f"{4*' '}MSE scores: {np.mean(scores_as_tensor):0.3f} +- {np.std(scores_as_tensor):0.3f}")
    
    with open('performances/scores.txt', 'a') as f:
        f.write("Hyperparameter Setting")
        f.write("----------------------")
        f.write(f"{4*' '}learning_rate: {learning_rate}")
        f.write(f"{4*' '}weight_decay : {weight_decay}")
        f.write(f"{4*' '}batch_size   : {batch_size}")
        f.write(f"{8*' '}MSE  : {np.mean(scores_as_tensor)} +- {np.std(scores_as_tensor)}")
        f.write(f"{8*' '}RMSE : {np.mean(torch.tensor(cv_rmse_scores).detach().cpu().numpy()):<10f} +- {np.std(torch.tensor(cv_rmse_scores).detach().cpu().numpy()):<10f}")
        f.write(f"{8*' '}MAE  : {np.mean(torch.tensor(cv_mae_scores).detach().cpu().numpy()):<10f} +- {np.std(torch.tensor(cv_mae_scores).detach().cpu().numpy()):<10f}")
        f.write(f"{8*' '}R2   : {np.mean(torch.tensor(cv_r2_scores).detach().cpu().numpy()):<10f} +- {np.std(torch.tensor(cv_r2_scores).detach().cpu().numpy()):<10f}")        
        f.write(f"{8*' '}PCC  : {np.mean(torch.tensor(cv_pcc_scores).detach().cpu().numpy()):<10f} +- {np.std(torch.tensor(cv_pcc_scores).detach().cpu().numpy()):<10f}")
        f.write(f"{8*' '}SCC  : {np.mean(torch.tensor(cv_scc_scores).detach().cpu().numpy()):<10f} +- {np.std(torch.tensor(cv_scc_scores).detach().cpu().numpy()):<10f}")        

    return np.mean(scores_as_tensor)
        

# -----------------------------------------------------------------------------
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
        logging.info(print_args_summary(args))
        
        # ---------------------------------------------------------------------        
        # Define hyperparameters to optimize for.
        param_space = [
            Real(name='learning_rate', low=0.0001, high=0.1, prior='log-uniform'),
            Real(name='weight_decay', low=0.000000001, high=0.001, prior='log-uniform'),
            Integer(name='batch_size', low=16, high=1024)#, prior='uniform')      
#             Real(name='dropout', low=0.0, high=0.5, prior='uniform'),            
            # Real(0.0, 0.5, prior='uniform', name='dropout'),
            # Real(0.0, 0.001, prior='log-uniform', name='weight_decay')
            # Dimension(low=16, high=1024, prior='uniform', name='batch_size')
            # Categorical(categories=['GCNConv', 'GATConv'], name='conv_type')
        ]

        train_val_set, test_set = train_test_split(
            drm, 
            test_size=args.test_ratio, #1 - (args.train_ratio + args.val_ratio),
            random_state=args.seed,
            stratify=drm['CELL_LINE_NAME']
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        
        # Define fixed parameters for the objective function.
        fixed_params = {
            'args': args,
            'device': device,
            'datasets': [train_val_set, test_set, cl_graphs, fingerprints_dict]
        }
        
        import skopt
        print(skopt.__version__)
        print(param_space)

        objective_gt_fixed = partial(objective_gt, **fixed_params)

        # ---------------------------------------------------------------------        
        # Run the Bayesian hyperparameter optimization.
        bayes_res = gbrt_minimize(
            objective_gt_fixed, 
            param_space, 
            n_calls=10, # TODO: make this higher
            random_state=args.seed,
            n_jobs=args.num_workers
        )
        optimal_params = bayes_res.x

        logging.info("Results of Bayesian Hyperparameter optimization")
        logging.info("===============================================")
        logging.info(f"{4*' '}Optimal hyperparameter values:")
        for param, value in zip(bayes_res.space, optimal_params):
            logging.info(f"{8*' '} {param.name:15s} optimal value: {value}")        
        
        # ---------------------------------------------------------------------
        # --- Train final model using the optimal hyperparameters ---
        optimal_learning_rate = optimal_params[0]
        
        train_val_loader, test_loader = create_gt_final_datasets(
            [train_val_set, test_set],
            cl_graphs, 
            fingerprints_dict, 
            args
        )
        
        logging.info("Finished creating pytorch training datasets!")
        logging.info("Number of batches per dataset:")
        logging.info(f"  train : {len(train_loader)}")
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
                                     lr=optimal_learning_rate)

        # Build the model.
        build_model = BuildGraphTabModel(
            model=model,
            criterion=loss_func,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            train_loader=train_loader,
            val_loader=train_loader,        
            test_loader=test_loader, 
            device=device
        )
        logging.info(build_model.model)      

        # Train the model.
        performance_stats = build_model.train(build_model.train_loader)
        
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
    }, PERFORMANCES + f'bayes_model_performance_{args.model}_{args.version}_{args.gdsc.lower()}_{args.combined_score_thresh}_{args.seed}_{args.file_ending}.pth')
        


if __name__ == "__main__":
    main()        
