import random
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

from argparse                      import ArgumentParser
from pathlib                       import Path
from config                        import PATH_SUMMARY_DATASETS
from src.models.GraphTab.graph_tab import GraphTabDataset, create_graph_tab_datasets, BuildGraphTabModel, GraphTab_v1
from src.models.TabGraph.tab_graph import TabGraphDataset, create_tab_graph_datasets, BuildTabGraphModel, TabGraph_v1
from src.preprocess.processor      import Processor
from sklearn.model_selection       import train_test_split
from torch_geometric.loader        import DataLoader


def parse_args():
    parser = ArgumentParser(description='GNNs for Drug Response Prediction in Cancer')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=2, help='the batch size (default: 10)')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.5, help='validation set ratio inside the test set (default: 0.5)')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs (default: )')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability (default: 0.1)')
    parser.add_argument('--model', type=str, default='GraphTab', 
                        help='name of the model to run, options: [`TabTab`, `GraphTab`, `TabGraph`, `GraphGraph`]')
    parser.add_argument('--version', type=str, default='v3', help='model version to run')
    parser.add_argument('--download', type=str, default='n', 
                        help="If raw data should be downloaded press either [`y`, `yes`, `1`]. " \
                           + "If no data should be downloaded press either [`n`, `no`, `0`]")
    parser.add_argument('--process', type=str, default='n', 
                        help="If data should be processed press either [`y`, `yes`, `1`]. " \
                           + "If no data should be processed press either [`n`, `no`, `0`]")   
    parser.add_argument('--raw_path', type=str, default='../data/raw/', help='path of the raw datasets')
    parser.add_argument('--processed_path', type=str, default='../data/processed/', help='path of the processed datasets')    
    
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

    # Initialize processor used for downloading and creation of training datasets.
    processor = Processor(raw_path=args.raw_path, 
                          processed_path=args.processed_path)    
    
    # Download data if necessary.
    if args.download in ['y', 'yes', '1']:
        processor.create_raw_datasets()
        
    # Created training datatsets if necessary.
    if args.process in ['y', 'yes', '1']:
        processor.create_processed_datasets()


if __name__ == "__main__":
    main()        
