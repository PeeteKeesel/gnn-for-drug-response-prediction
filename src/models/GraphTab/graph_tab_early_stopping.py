import logging
import time
import torch
import torch.nn as nn
import numpy    as np

from torch_geometric.data    import Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader  import DataLoader as PyG_DataLoader
from torch_geometric.nn      import Sequential, GCNConv, GATConv, global_mean_pool, global_max_pool
from tqdm                    import tqdm
from time                    import sleep
from sklearn.metrics         import r2_score, mean_absolute_error
from scipy.stats             import pearsonr, spearmanr
from ignite.engine           import Engine, Events
from ignite.handlers         import EarlyStopping
from functools               import partial


torch.multiprocessing.set_sharing_strategy('file_system')


class GraphTabDataset(Dataset): 
    def __init__(self, cl_graphs, drugs, drug_response_matrix):
        super().__init__()

        # SMILES fingerprints of the drugs and cell-line graphs.
        self.drugs = drugs
        self.cell_line_graphs = cl_graphs

        # Lookup datasets for the response values.
        drug_response_matrix.reset_index(drop=True, inplace=True)
        self.cell_lines = drug_response_matrix['CELL_LINE_NAME']
        self.drug_ids = drug_response_matrix['DRUG_ID']
        self.drug_names = drug_response_matrix['DRUG_NAME']
        self.ic50s = drug_response_matrix['LN_IC50']

    def __len__(self):
        return len(self.ic50s)

    def __getitem__(self, idx: int):
        """
        Returns a tuple of cell-line, drug and the corresponding ln(IC50)
        value for a given index.

        Args:
            idx (`int`): Index to specify the row in the drug response matrix.  
        Returns:
            `Tuple[torch_geometric.data.data.Data, np.ndarray, np.float64]`:
            Tuple of a cell-line graph, drug SMILES fingerprint and the 
            corresponding ln(IC50) value.
        """
        assert self.cell_lines.iloc[idx] in self.cell_line_graphs.keys(), \
            f"Didn't find CELLLINE: {self.cell_line_graphs.iloc[idx]}"        
        assert self.drug_ids.iloc[idx] in self.drugs.keys(), \
            f"Didn't find DRUG: {self.drug_ids.iloc[idx]}"
        return (self.cell_line_graphs[self.cell_lines.iloc[idx]], 
                self.drugs[self.drug_ids.iloc[idx]],
                self.ic50s.iloc[idx])

    def print_dataset_summary(self):
        logging.info(f"GraphTabDataset Summary")
        logging.info(f"{23*'='}")
        logging.info(f"# observations : {len(self.ic50s)}")
        logging.info(f"# cell-lines   : {len(np.unique(self.cell_lines))}")
        logging.info(f"# drugs        : {len(np.unique(self.drug_names))}")
        logging.info(f"# genes        : {self.cell_line_graphs[next(iter(self.cell_line_graphs))].x.shape[0]}")

def create_gt_loaders(
        drm_train,
        drm_test,
        cl_graphs,
        drug_mat,
        args
    ):
    """Create train and test pytorch.DataLoaders for outer k-fold cross validation."""
    logging.info(f"{8*' '}train shape: {drm_train.shape}")   
    logging.info(f"{8*' '}test  shape: {drm_test.shape}")
    
    train_dataset = GraphTabDataset(
        cl_graphs=cl_graphs,
        drugs=drug_mat,
        drug_response_matrix=drm_train
    )
    test_dataset = GraphTabDataset(
        cl_graphs=cl_graphs,
        drugs=drug_mat,
        drug_response_matrix=drm_test
    )
    
    logging.info(f"{8*' '}train_dataset:"); train_dataset.print_dataset_summary()   
    logging.info(f"{8*' '}test_dataset :"); test_dataset.print_dataset_summary()
    
    train_loader = PyG_DataLoader(
        dataset=train_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=True, 
        num_workers=args.num_workers
    )
    test_loader = PyG_DataLoader(
        dataset=test_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader
    
    
def create_graph_tab_datasets(drm, cl_graphs, drug_mat, args):
    logging.info(f"Full     shape: {drm.shape}")
    train_set, test_val_set = train_test_split(drm, 
                                               test_size=args.TEST_VAL_RATIO, 
                                               random_state=args.RANDOM_SEED,
                                               stratify=drm['CELL_LINE_NAME'])
    test_set, val_set = train_test_split(test_val_set,
                                         test_size=args.VAL_RATIO,
                                         random_state=args.RANDOM_SEED,
                                         stratify=test_val_set['CELL_LINE_NAME'])
    logging.info(f"train    shape: {train_set.shape}")
    logging.info(f"test_val shape: {test_val_set.shape}")
    logging.info(f"test     shape: {test_set.shape}")
    logging.info(f"val      shape: {val_set.shape}")

    train_dataset = GraphTabDataset(cl_graphs=cl_graphs, drugs=drug_mat, drug_response_matrix=train_set)
    test_dataset = GraphTabDataset(cl_graphs=cl_graphs, drugs=drug_mat, drug_response_matrix=test_set)
    val_dataset = GraphTabDataset(cl_graphs=cl_graphs, drugs=drug_mat, drug_response_matrix=val_set)

    logging.info("train_dataset:")
    train_dataset.print_dataset_summary()
    logging.info("test_dataset:")
    test_dataset.print_dataset_summary()
    logging.info("val_dataset:")
    val_dataset.print_dataset_summary()

    # TODO: try out different `num_workers`.
    train_loader = PyG_DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)
    test_loader = PyG_DataLoader(dataset=test_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)
    val_loader = PyG_DataLoader(dataset=val_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)

    return train_loader, test_loader, val_loader    


class BuildGraphTabModel(Engine):
    def __init__(self, model, criterion, optimizer, num_epochs, 
        train_loader, test_loader, 
        early_stopping_threshold, device):
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.train_loader = train_loader
        self.test_loader = test_loader
#         self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopping_threshold = early_stopping_threshold
        self.device = device
        
#         self.evaluator = Engine(partial(self.validate, loader=self.test_loader))
        
#         super(BuildGraphTabModel, self).__init__(
#             process_function=self.validate
#         )        
        
#     def _score_function(self, engine):
#         return -engine.state.metrics['mse']

    def train(self, loader): 
        train_epoch_losses, test_epoch_losses = [], []
        train_epoch_rmse, test_epoch_rmse = [], []
        train_epoch_mae, test_epoch_mae = [], []
        train_epoch_r2, test_epoch_r2 = [], []
        train_epoch_pcc, test_epoch_pcc = [], []
        train_epoch_scc, test_epoch_scc = [], []      
        train_epoch_time = []
        all_batch_losses = [] # TODO: this is just for monitoring
        n_batches = len(loader)
        
#         # Early stopping.
#         early_stopping = EarlyStopping(
#             patience=self.early_stopping_threshold,
#             score_function=self._score_function,
#             trainer=self
#         )
#         self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
        early_stopping_counter = 0
        early_stopped_epoch = self.num_epochs
        best_loss = float('inf')

        # Iterate through epochs.
        self.model = self.model.float()
        for epoch in range(1, self.num_epochs+1):
            tic = time.time()
            self.model.train()
            batch_losses = []
            y_true, y_pred = [], []
            for i, data in enumerate(tqdm(loader, desc='Iteration (train)')):
                sleep(0.01)
                cell, drug, ic50s = data
                drug = torch.stack(drug, 0).transpose(1, 0) # Note that this is only neede when geometric 
                                                            # Dataloader is used and no collate.
                cell, drug, ic50s = cell.to(self.device), drug.to(self.device), ic50s.to(self.device)

                self.optimizer.zero_grad()

                # Models predictions of the ic50s for a batch of cell-lines and drugs
                preds = self.model(cell.x.float(),
                                   cell.edge_index,
                                   cell.batch,
                                   drug.float()).unsqueeze(1)
                loss = self.criterion(preds, ic50s.view(-1, 1).float()) # =train_loss
                batch_losses.append(loss)

                y_true.append(ic50s.view(-1, 1))
                y_pred.append(preds)             

                loss.backward()
                self.optimizer.step()

            all_batch_losses.append(batch_losses)
            total_epoch_loss = sum(batch_losses)
            train_epoch_losses.append(total_epoch_loss / n_batches)

            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            train_mse = train_epoch_losses[-1]
            train_epoch_rmse.append(torch.sqrt(train_mse).item())
            train_epoch_mae.append(mean_absolute_error(y_true.detach().cpu(), 
                                                       y_pred.detach().cpu()))
            train_epoch_r2.append(r2_score(y_true.detach().cpu(), 
                                           y_pred.detach().cpu()))
            train_epoch_pcc.append(pearsonr(y_true.detach().cpu().numpy().flatten(), 
                                            y_pred.detach().cpu().numpy().flatten()).statistic)
            train_epoch_scc.append(spearmanr(y_true.detach().cpu().numpy().flatten(),
                                             y_pred.detach().cpu().numpy().flatten()).statistic)             
                     
            # Validate the model.
            mse, rmse, mae, r2, pcc, scc, _, _  = self.validate(self.test_loader)
            test_epoch_losses.append(mse)
            test_epoch_rmse.append(rmse)
            test_epoch_mae.append(mae)
            test_epoch_r2.append(r2)
            test_epoch_pcc.append(pcc)
            test_epoch_scc.append(scc)
            
            train_epoch_time.append(time.time() - tic)            
            
            logging.info(f"===Epoch {epoch:03.0f}===")
            logging.info(f"Train | MSE: {train_mse:2.5f} | RMSE: {train_epoch_rmse[-1]:2.5f} | MAE: {train_epoch_mae[-1]:2.5f} | R2: {train_epoch_r2[-1]:2.5f} | PCC: {train_epoch_pcc[-1]:2.5f} | SCC: {train_epoch_scc[-1]:2.5f}")
            logging.info(f"Test  | MSE: {mse:2.5f} | RMSE: {test_epoch_rmse[-1]:2.5f} | MAE: {test_epoch_mae[-1]:2.5f} | R2: {test_epoch_r2[-1]:2.5f} | PCC: {test_epoch_pcc[-1]:2.5f} | SCC: {test_epoch_scc[-1]:2.5f}")
            
            # Check early stopping criteria.
            if mse < best_loss:
                best_loss = mse
                early_stopping_counter = 0 
            else: 
                early_stopping_counter += 1
                
            if early_stopping_counter >= self.early_stopping_threshold:  
                logging.info("EarlyStopping: Stop training!")
                logging.info(f"{4*' '}Stopped at epoch {epoch}")
                early_stopped_epoch = epoch
                break
            
#         # Final model performance.
        logging.info(f"===Epoch {epoch:03.0f}===")
        logging.info(f"Train | MSE: {train_epoch_losses[-1]:2.5f} | RMSE: {train_epoch_rmse[-1]} | MAE: {train_epoch_mae[-1]} | R2: {train_epoch_r2[-1]} | PCC: {train_epoch_pcc[-1]} | SCC: {train_epoch_scc[-1]}")
        logging.info(f"Test  | MSE: {test_epoch_losses[-1]:2.5f} | RMSE: {test_epoch_rmse[-1]} | MAE: {test_epoch_mae[-1]} | R2: {test_epoch_r2[-1]} | PCC: {test_epoch_pcc[-1]} | SCC: {test_epoch_scc[-1]}")           

        performance_stats = {
            'train': {
                'mse': train_epoch_losses,
                'rmse': train_epoch_rmse,
                'mae': train_epoch_mae,
                'r2': train_epoch_r2,
                'pcc': train_epoch_pcc,
                'scc': train_epoch_scc,                
                'epoch_times': train_epoch_time,
                'early_stopped_epoch': early_stopped_epoch
            },
            'test': {
                'mse': test_epoch_losses,
                'rmse': test_epoch_rmse,
                'mae': test_epoch_mae,
                'r2': test_epoch_r2,
                'pcc': test_epoch_pcc,
                'scc': test_epoch_scc                
            }          
        }

        return performance_stats           

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(loader, desc='Iteration (val)'):
                sleep(0.01)
                cl, dr, ic50 = data
                dr = torch.stack(dr, 0).transpose(1, 0)
                cl, dr, ic50 = cl.to(self.device), dr.to(self.device), ic50.to(self.device)

                preds = self.model(cl.x.float(), 
                                   cl.edge_index, 
                                   cl.batch, 
                                   dr.float()).unsqueeze(1)
#                 ic50 = ic50.to(self.device)
                total_loss += self.criterion(preds, ic50.view(-1,1).float())
                # total_loss += F.mse_loss(preds, ic50.view(-1, 1).float(), reduction='sum')
                y_true.append(ic50.view(-1, 1))
                y_pred.append(preds)
        
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        
        # Calculate performance metrics.        
        mse = total_loss / len(loader)
        rmse = torch.sqrt(mse)
        mae = mean_absolute_error(y_true.detach().cpu(), 
                                  y_pred.detach().cpu())
        r2 = r2_score(y_true.detach().cpu(), 
                      y_pred.detach().cpu())
        pcc, _ = pearsonr(y_true.detach().cpu().numpy().flatten(), 
                          y_pred.detach().cpu().numpy().flatten())
        scc, _ = spearmanr(y_true.detach().cpu().numpy().flatten(), 
                           y_pred.detach().cpu().numpy().flatten())        

        return mse, rmse, mae, r2, pcc, scc, y_true, y_pred    
    
"""
GraphTab model using 
    - GCNConv
    - global mean pooling
    - no Batch normalization between the GCNConv layers
References:
    - https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    - https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.sequential.Sequential
"""
class GraphTab_v1(torch.nn.Module):
    def __init__(self, dropout):
        super(GraphTab_v1, self).__init__()

        # Cell-line graph branch. Obtains node embeddings.
        self.cell_emb = Sequential('x, edge_index, batch', 
            [
                (GCNConv(in_channels=4, out_channels=256), 'x, edge_index -> x1'), # TODO: GATConv() vs GCNConv()
                nn.ReLU(inplace=True),
                (GCNConv(in_channels=256, out_channels=256), 'x1, edge_index -> x2'),
                nn.ReLU(inplace=True),
                (global_mean_pool, 'x2, batch -> x3'), 
                # Start embedding
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(128, 128),
                nn.ReLU()
            ]
        )

        self.drug_emb = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()          
        )

        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

    def forward(self, cell_x, cell_edge_index, cell_batch, drug):
        drug_emb = self.drug_emb(drug)
        cell_emb = self.cell_emb(cell_x, cell_edge_index, cell_batch)
        concat = torch.cat([cell_emb, drug_emb], -1)
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred        
    
      
"""
GraphTab model using 
    - GATConv
    - global max pooling
    - no batch normalization between the GCNConv layers
References:
    - https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
    - https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.sequential.Sequential
"""
class GraphTab(torch.nn.Module):
    def __init__(self, 
                 dropout, 
                 conv_type='GCNConv', 
                 conv_layers=2, 
                 global_pooling='max'):
        super(GraphTab, self).__init__()

        # Note: in_channels = number of features.
        if global_pooling == 'max':
            if conv_type == 'GCNConv':
                if conv_layers == 2:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GCNConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
    #                         nn.Dropout(p=dropout),
                            (GCNConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                
                            (global_max_pool, 'x2, batch -> x3'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )
                elif conv_layers == 3:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GCNConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            (GCNConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True), 
                            (GCNConv(in_channels=128, out_channels=128), 'x2, edge_index -> x3'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                        
                            (global_max_pool, 'x3, batch -> x4'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )                
            elif conv_type == 'GATConv':
                if conv_layers == 2:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GATConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
    #                         nn.Dropout(p=dropout),
                            (GATConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                
                            (global_max_pool, 'x2, batch -> x3'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )
                elif conv_layers == 3:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GATConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            (GATConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            (GATConv(in_channels=128, out_channels=128), 'x2, edge_index -> x3'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                
                            (global_max_pool, 'x3, batch -> x4'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )
        elif global_pooling == 'mean':
            if conv_type == 'GCNConv':
                if conv_layers == 2:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GCNConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
    #                         nn.Dropout(p=dropout),
                            (GCNConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                
                            (global_mean_pool, 'x2, batch -> x3'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )
                elif conv_layers == 3:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GCNConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            (GCNConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True), 
                            (GCNConv(in_channels=128, out_channels=128), 'x2, edge_index -> x3'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                        
                            (global_mean_pool, 'x3, batch -> x4'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )                
            elif conv_type == 'GATConv':
                if conv_layers == 2:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GATConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
    #                         nn.Dropout(p=dropout),
                            (GATConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                
                            (global_mean_pool, 'x2, batch -> x3'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )
                elif conv_layers == 3:
                    self.cell_emb = Sequential('x, edge_index, batch', 
                        [
                            (GATConv(in_channels=4, out_channels=128), 'x, edge_index -> x1'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            (GATConv(in_channels=128, out_channels=128), 'x1, edge_index -> x2'),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            (GATConv(in_channels=128, out_channels=128), 'x2, edge_index -> x3'),
                            nn.BatchNorm1d(128),                        
    #                         nn.ReLU(inplace=True),                
                            (global_mean_pool, 'x3, batch -> x4'), 
                            nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(128, 128),
                            nn.ReLU()
                        ]
                    )                    
        else:
            raise ValueError(
                f"Input conv_type `{conv_type}` has not been implemented! Choose out of [`GCNConv`, `GATConv`]"
            )

        self.drug_emb = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()          
        )

        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, cell_x, cell_edge_index, cell_batch, drug):
        drug_emb = self.drug_emb(drug)
#         print(f"""Input
#         drug            : {drug.size()}
#         cell.x          : {cell_x.size()}
#         cell.edge_index : {cell_edge_index.size()}
#         cell.batch      : {cell_batch.size()}
#         """)
        cell_emb = self.cell_emb(cell_x, cell_edge_index, cell_batch)
#         print(f"drug_emb.size: {drug_emb.size()}")
#         print(f"cell_emb.size: {cell_emb.size()}")
        concat = torch.cat([cell_emb, drug_emb], -1)
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred      