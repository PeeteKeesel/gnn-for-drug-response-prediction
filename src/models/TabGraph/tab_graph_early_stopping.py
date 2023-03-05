import logging
import time
import torch
import torch.nn as nn
import numpy    as np

from torch_geometric.data    import Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader  import DataLoader as PyG_DataLoader
from torch_geometric.nn      import Sequential, GINConv
from torch_geometric.nn      import global_mean_pool, global_max_pool, global_add_pool
from tqdm                    import tqdm
from time                    import sleep
from sklearn.metrics         import r2_score, mean_absolute_error
from scipy.stats             import pearsonr, spearmanr
from ignite.engine           import Engine, Events
from ignite.handlers         import EarlyStopping
from functools               import partial


torch.multiprocessing.set_sharing_strategy('file_system')


class TabGraphDataset(Dataset): 
    def __init__(self, 
                 cl_gene_mat, 
                 drug_graphs, 
                 drm):
        super().__init__()

        # Cell-line gene matrix and drug SMILES fingerprints graphs.
        self.cl_gene_mat = cl_gene_mat
        self.drug_graphs = drug_graphs

        # Lookup datasets for the response values.
        drm.reset_index(drop=True, inplace=True)
        self.cell_lines = drm['CELL_LINE_NAME']
        self.drug_ids = drm['DRUG_ID']
        self.drug_names = drm['DRUG_NAME']
        self.ic50s = drm['LN_IC50']

    def __len__(self):
        return len(self.ic50s)

    def __getitem__(self, idx: int):
        """
        Returns a tuple of cell-line, drug and the corresponding ln(IC50)
        value for a given index.

        Args:
            idx (`int`): Index to specify the row in the drug response matrix.  
        Returns:
            `np.ndarray, Tuple[torch_geometric.data.data.Data], np.float64]`:
            Tuple of a cell-line gene values, drug SMILES fingerprint graph and 
            the corresponding ln(IC50) value.
        """
        return (self.cl_gene_mat.loc[self.cell_lines.iloc[idx]].values.tolist(),
                self.drug_graphs[self.drug_ids.iloc[idx]],
                self.ic50s.iloc[idx])

    def print_dataset_summary(self):
        logging.info(f"TabGraphDataset Summary")
        logging.info(f"{23*'='}")
        logging.info(f"# observations : {len(self.ic50s)}")
        logging.info(f"# cell-lines   : {len(np.unique(self.cell_lines))}")
        logging.info(f"# drugs        : {len(self.drug_graphs.keys())}")
        logging.info(f"# genes        : {len(self.cl_gene_mat.columns)/4}")

def create_tg_loaders(
        drm_train,
        drm_test,
        cl_gene_mat,
        drug_graphs,
        args
    ):
    """Create train and test pytorch.DataLoaders for outer k-fold cross validation."""
    logging.info(f"{8*' '}train shape: {drm_train.shape}")   
    logging.info(f"{8*' '}test  shape: {drm_test.shape}")
    
    train_dataset = TabGraphDataset(
        cl_gene_mat=cl_gene_mat,
        drug_graphs=drug_graphs,
        drm=drm_train
    )
    test_dataset = TabGraphDataset(
        cl_gene_mat=cl_gene_mat,
        drug_graphs=drug_graphs,
        drm=drm_test
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


class BuildTabGraphModel(Engine):
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 num_epochs, 
                 train_loader, 
                 test_loader, 
                 early_stopping_threshold, 
                 device):
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopping_threshold = early_stopping_threshold
        self.device = device

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
                cell = torch.stack(cell, 0).transpose(1, 0) # Note that this is only neede when geometric 
                                                            # Dataloader is used and no collate.
                cell, drug, ic50s = cell.to(self.device), drug.to(self.device), ic50s.to(self.device)

                self.optimizer.zero_grad()

                # Models predictions of the ic50s for a batch of cell-lines and drugs
                preds = self.model(cell.float(),
                                   drug.x.float(),
                                   drug.edge_index,
                                   drug.batch).unsqueeze(1)
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
            train_epoch_rmse.append(torch.sqrt(train_mse))
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
            
        logging.info(f"===Epoch {epoch:03.0f}===")
        logging.info(f"Train | MSE: {train_epoch_losses[-1]:2.5f} | RMSE: {train_epoch_rmse[-1]} | MAE: {train_epoch_mae[-1]} | R2: {train_epoch_r2[-1]} | PCC: {train_epoch_pcc[-1]} | SCC: {train_epoch_scc[-1]}")
        logging.info(f"Test  | MSE: {test_epoch_losses[-1]:2.5f} | RMSE: {test_epoch_rmse[-1]} | MAE: {test_epoch_mae[-1]} | R2: {test_epoch_r2[-1]} | PCC: {test_epoch_pcc[-1]} | SCC: {test_epoch_scc[-1]}")
            
#         # Final model performance.
#         mse_te, rmse_te, mae_te, r2_te, pcc_te, scc_te, _, _ = self.validate(self.test_loader)
#         logging.info(f"Test       | MSE: {mse_te:2.5f}")            

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
                cl = torch.stack(cl, 0).transpose(1, 0)
                cl, dr, ic50 = cl.to(self.device), dr.to(self.device), ic50.to(self.device)

                preds = self.model(cl.float(), 
                                   dr.x.float(),
                                   dr.edge_index,
                                   dr.batch).unsqueeze(1)
                ic50 = ic50.to(self.device)
                total_loss += self.criterion(preds, ic50.view(-1,1).float())
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
    

class TabGraph(torch.nn.Module):
    def __init__(self, 
                 cell_inp_dim: int,
                 dropout: float):
        super(TabGraph, self).__init__()

        self.cell_emb = nn.Sequential(
            # TODO: the 2784 needs to be an arguments! provide it in the main!
            nn.Linear(cell_inp_dim, 516),
            nn.BatchNorm1d(516),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(516, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()         
        )

        self.drug_emb = Sequential('x, edge_index, batch', 
            [
                (GINConv(
                    nn.Sequential(
                        nn.Linear(9, 128), # 9 = num_node_features
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 128)
                    )
                ), 'x, edge_index -> x1'),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                (GINConv(
                    nn.Sequential(
                        nn.Linear(128, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 128)
                    )
                ), 'x1, edge_index -> x2'),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                # TODO: research maybe JumpingKnowledge at this point
                (global_max_pool, 'x2, batch -> x3'),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 128),
                nn.ReLU()
            ]
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


    
    def forward(self, 
                cell, 
                drug_x, 
                drug_edge_index,
                drug_batch):
        cell_emb = self.cell_emb(cell)
        drug_emb = self.drug_emb(drug_x, drug_edge_index, drug_batch)
        concat = torch.cat([cell_emb, drug_emb], -1)
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred      