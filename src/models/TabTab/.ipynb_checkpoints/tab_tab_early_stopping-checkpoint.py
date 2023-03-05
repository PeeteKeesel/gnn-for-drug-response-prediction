import logging
import time
import torch
import torch.nn as nn
import numpy    as np

from torch_geometric.data    import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data        import DataLoader
from tqdm                    import tqdm
from time                    import sleep
from sklearn.metrics         import r2_score, mean_absolute_error
from scipy.stats             import pearsonr, spearmanr
from ignite.engine           import Engine, Events
from ignite.handlers         import EarlyStopping


from functools import partial


class TabTabDataset(Dataset): 
    def __init__(self, cl_mat, drug_mat, drm):
        super().__init__()
        self.cl_mat = cl_mat
        self.drug_mat = drug_mat

        drm.reset_index(drop=True, inplace=True)
        self.cls = drm['CELL_LINE_NAME']
        self.drug_ids = drm['DRUG_ID']
        self.drug_names = drm['DRUG_NAME']
        self.ic50s = drm['LN_IC50']

    def __len__(self):
        return len(self.ic50s)

    def __getitem__(self, idx: int):
        """
        Returns a tuple of cell-line-gene features, drug smiles fingerprints 
        and the corresponding ln(IC50) values for a given index.

        Args:
            idx (`int`): Index to specify the row in the drug response matrix.  
        Returns
            `Tuple[np.ndarray, np.ndarray, np.float64]]`: Tuple of cell-line 
                gene feature values, drug SMILES fingerprints and the 
                corresponding ln(IC50) target values.
        """  
        return (self.cl_mat.loc[self.cls.iloc[idx]], 
                self.drug_mat.loc[self.drug_ids.iloc[idx]],
                self.ic50s.iloc[idx])

    def print_dataset_summary(self):
        logging.info(f"TabTabDataset Summary")
        logging.info(21*'=')
        logging.info(f"# observations : {len(self.ic50s)}")
        logging.info(f"# cell-lines   : {len(np.unique(self.cls))}")
        logging.info(f"# drugs        : {len(np.unique(self.drug_names))}")
        logging.info(f"# genes        : {len([col for col in self.cl_mat.columns[1:] if '_cnvg' in col])}")


def _collate_tab_tab(samples):
    cell_lines, drugs, ic50s = map(list, zip(*samples))
    cell_lines = [torch.tensor(cl, dtype=torch.float64) for cl in cell_lines]
    drugs = [torch.tensor(drug, dtype=torch.float64) for drug in drugs]
    
    return torch.stack(cell_lines, 0), torch.stack(drugs, 0), torch.tensor(ic50s)


def create_tt_loaders(
        drm_train,
        drm_test,
        cl_mat, 
        drug_mat,
        args
    ):
    """Create train and test pytorch.DataLoaders for outer k-fold cross validation."""
    
    logging.info(f"{8*' '}train shape: {drm_train.shape}")   
    logging.info(f"{8*' '}test  shape: {drm_test.shape}")    
    
    train_dataset = TabTabDataset(
        cl_mat=cl_mat, 
        drug_mat=drug_mat, 
        drm=drm_train
    )
    test_dataset = TabTabDataset(
        cl_mat=cl_mat, 
        drug_mat=drug_mat, 
        drm=drm_test
    )
    
    logging.info(f"{8*' '}train_dataset:"); train_dataset.print_dataset_summary()   
    logging.info(f"{8*' '}test_dataset :"); test_dataset.print_dataset_summary()
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=True, 
        collate_fn=_collate_tab_tab,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=True, 
        collate_fn=_collate_tab_tab,
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader


    
def create_tab_tab_datasets(drm, cl_mat, drug_mat, args):
    train_set, test_val_set = train_test_split(drm, 
                                               test_size=args.TEST_VAL_RATIO,
                                               random_state=args.RANDOM_SEED,
                                               stratify=drm['CELL_LINE_NAME'])
    test_set, val_set = train_test_split(test_val_set, 
                                         test_size=args.VAL_RATIO, 
                                         random_state=args.RANDOM_SEED,
                                         stratify=test_val_set['CELL_LINE_NAME'])

    logging.info(f"train_set.shape: {train_set.shape}")
    logging.info(f"test_set.shape: {test_set.shape}")
    logging.info(f"val_set.shape: {val_set.shape}")

    train_dataset = TabTabDataset(cl_mat=cl_mat, 
                                  drug_mat=drug_mat, 
                                  drm=train_set)
    test_dataset = TabTabDataset(cl_mat=cl_mat, 
                                 drug_mat=drug_mat, 
                                 drm=test_set)
    val_dataset = TabTabDataset(cl_mat=cl_mat, 
                                drug_mat=drug_mat, 
                                drm=val_set)

    logging.info("train_dataset"); train_dataset.print_dataset_summary()
    logging.info("test_dataset"); test_dataset.print_dataset_summary()
    logging.info("val_dataset"); val_dataset.print_dataset_summary()

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.BATCH_SIZE, 
                              shuffle=True, 
                              collate_fn=_collate_tab_tab, 
                              num_workers=args.NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.BATCH_SIZE, 
                             shuffle=True, 
                             collate_fn=_collate_tab_tab, 
                             num_workers=args.NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=_collate_tab_tab, 
                            num_workers=args.NUM_WORKERS)  

    return train_loader, test_loader, val_loader          


class BuildTabTabModel(Engine):
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
        
        self.y_true, self.y_pred = [], []
        
        self.evaluator = Engine(partial(self.validate, loader=self.test_loader))
        
        super(BuildTabTabModel, self).__init__(
            process_function=self.validate
        )
        
    def _score_function(self, engine):
        return -engine.state.metrics['mse']        
    
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
        
        # Early stopping.
        early_stopping = EarlyStopping(
            patience=self.early_stopping_threshold,
            score_function=self._score_function,
            trainer=self
        )
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
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
                cell, drug, ic50s = cell.to(self.device), drug.to(self.device), ic50s.to(self.device)

                self.optimizer.zero_grad()

                # Models predictions of the ic50s for a batch of cell-lines and drugs
                preds = self.model(cell.float(), drug.float()).unsqueeze(1)
                loss = self.criterion(preds, ic50s.view(-1, 1).float()) # =train_loss
                batch_losses.append(loss)

                y_true.append(ic50s.view(-1, 1))
                y_pred.append(preds)             

                loss.backward()
                self.optimizer.step()

            all_batch_losses.append(batch_losses) # TODO: this is just for monitoring
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
                                            y_pred.detach().cpu().numpy().flatten()))
            train_epoch_scc.append(spearmanr(y_true.detach().cpu().numpy().flatten(),
                                             y_pred.detach().cpu().numpy().flatten()))            

            self.evaluator.state.metrics["total_loss"] = 0
            self.y_true, self.y_pred = [], []        
            self.evaluator.run(self.test_loader)             
    
            mse = self.evaluator.state.metrics['mse']
            rmse = self.evaluator.state.metrics['rmse']
            mae = self.evaluator.state.metrics['mae']
            r2 = self.evaluator.state.metrics['r2']
            pcc = self.evaluator.state.metrics['pcc']
            scc = self.evaluator.state.metrics['scc']
    
            test_epoch_losses.append(mse)
            test_epoch_rmse.append(rmse)
            test_epoch_mae.append(mae)
            test_epoch_r2.append(r2)
            test_epoch_pcc.append(pcc)
            test_epoch_scc.append(scc)
            
            train_epoch_time.append(time.time() - tic)

            logging.info(f"=====Epoch {epoch}")
            logging.info(f"Train | MSE: {train_mse:2.5f}")
            logging.info(f"Test  | MSE: {mse:2.5f}")
            
            # Check early stopping criteria
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
    def validate(self, engine, data, loader=None):
        self.model.eval()
#         print(f"engine: {engine}")
#         print(f"loader: {len(loader)}")
#         print(f"len(loader): {len(data)}")
#         y_true, y_pred = [], []
        total_loss = engine.state.metrics["total_loss"]
        with torch.no_grad():
            cl, dr, ic50 = data 
            cl, dr, ic50 = cl.to(self.device), dr.to(self.device), ic50.to(self.device)

            preds = self.model(cl.float(), dr.float()).unsqueeze(1)
            ic50 = ic50.to(self.device)
            total_loss = self.criterion(preds, ic50.view(-1,1).float())
            
            engine.state.metrics["total_loss"] += total_loss
            
            self.y_true.append(ic50.view(-1, 1))
            self.y_pred.append(preds)
            
#         with torch.no_grad():        
#             for data in tqdm(loader, desc='Iteration (val)'):
#                 sleep(0.01)
#                 print(f"data: {data}")
#                 print(f"size: {data.size()}")                
#                 cl, dr, ic50 = data
#                 cl, dr, ic50 = cl.to(self.device), dr.to(self.device), ic50.to(self.device)

#                 preds = self.model(cl.float(), dr.float()).unsqueeze(1)
#                 ic50 = ic50.to(self.device)
#                 total_loss += self.criterion(preds, ic50.view(-1,1).float())
#                 y_true.append(ic50.view(-1, 1))
#                 y_pred.append(preds)
        
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        
        # Calculate performance metrics.
        mse = engine.state.metrics["total_loss"] / len(loader)
        rmse = torch.sqrt(mse)
        mae = mean_absolute_error(y_true.detach().cpu(), 
                                  y_pred.detach().cpu())
        r2 = r2_score(y_true.detach().cpu(), 
                      y_pred.detach().cpu())
        pcc, _ = pearsonr(y_true.detach().cpu().numpy().flatten(), 
                          y_pred.detach().cpu().numpy().flatten())
        scc, _ = spearmanr(y_true.detach().cpu().numpy().flatten(), 
                           y_pred.detach().cpu().numpy().flatten()) 
        
        engine.state.metrics['mse'] = mse
        engine.state.metrics['rmse'] = rmse
        engine.state.metrics['mae'] = mae
        engine.state.metrics['r2'] = r2
        engine.state.metrics['pcc'] = pcc
        engine.state.metrics['scc'] = scc    

#         return mse, rmse, mae, r2, pcc, scc, y_true, y_pred


class TabTab(torch.nn.Module):
    def __init__(self, cell_inp_dim, dropout):
        super(TabTab, self).__init__()

        self.cell_emb = nn.Sequential(
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

    def forward(self, cell, drug):
        drug_emb = self.drug_emb(drug)
        cell_emb = self.cell_emb(cell)
        concat = torch.cat([cell_emb, drug_emb], -1)
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred        