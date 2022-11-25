import torch
import torch.nn as nn
import numpy    as np

from torch_geometric.data    import Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader  import DataLoader as PyG_DataLoader
from torch_geometric.nn      import Sequential, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn      import Sequential, GINConv, global_mean_pool, global_max_pool
from tqdm                    import tqdm
from time                    import sleep
from sklearn.metrics         import r2_score, mean_absolute_error
from scipy.stats             import pearsonr


class TabGraphDataset(Dataset): 
    def __init__(self, cl_gene_mat, drug_graphs, drm):
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
        print(f"TabGraphDataset Summary")
        print(f"{23*'='}")
        print(f"# observations : {len(self.ic50s)}")
        print(f"# cell-lines   : {len(np.unique(self.cell_lines))}")
        print(f"# drugs        : {len(self.drug_graphs.keys())}")
        print(f"# genes        : {len(self.cl_gene_mat.columns)/4}")


def create_tab_graph_datasets(drm, cl_gene_mat, drug_graphs, args):
    print(f"Full     shape: {drm.shape}")
    train_set, test_val_set = train_test_split(drm, 
                                               test_size=args.TEST_VAL_RATIO, 
                                               random_state=args.RANDOM_SEED,
                                               stratify=drm['CELL_LINE_NAME'])
    test_set, val_set = train_test_split(test_val_set,
                                         test_size=args.VAL_RATIO,
                                         random_state=args.RANDOM_SEED,
                                         stratify=test_val_set['CELL_LINE_NAME'])
    print(f"train    shape: {train_set.shape}")
    print(f"test_val shape: {test_val_set.shape}")
    print(f"test     shape: {test_set.shape}")
    print(f"val      shape: {val_set.shape}")

    train_dataset = TabGraphDataset(cl_gene_mat=cl_gene_mat, drug_graphs=drug_graphs, drm=train_set)
    test_dataset = TabGraphDataset(cl_gene_mat=cl_gene_mat, drug_graphs=drug_graphs, drm=test_set)
    val_dataset = TabGraphDataset(cl_gene_mat=cl_gene_mat, drug_graphs=drug_graphs, drm=val_set)

    print("\ntrain_dataset:")
    train_dataset.print_dataset_summary()
    print("\n\ntest_dataset:")
    test_dataset.print_dataset_summary()
    print("\n\nval_dataset:")
    val_dataset.print_dataset_summary()

    # TODO: try out different `num_workers`.
    train_loader = PyG_DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)
    test_loader = PyG_DataLoader(dataset=test_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)
    val_loader = PyG_DataLoader(dataset=val_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)

    return train_loader, test_loader, val_loader    


class BuildTabGraphModel():
    def __init__(self, model, criterion, optimizer, num_epochs, 
        train_loader, test_loader, val_loader, device):
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, loader): 
        train_epoch_losses, val_epoch_losses = [], []
        train_epoch_rmse, val_epoch_rmse = [], []
        train_epoch_mae, val_epoch_mae = [], []
        train_epoch_r2, val_epoch_r2 = [], []
        train_epoch_pcorr, val_epoch_pcorr = [], []
        all_batch_losses = [] # TODO: this is just for monitoring
        n_batches = len(loader)

        self.model = self.model.float() # TODO: maybe remove
        for epoch in range(self.num_epochs):
            self.model.train()
            batch_losses = []
            y_true, y_pred = [], []
            for i, data in enumerate(tqdm(loader, desc='Iteration (Train)')):
                sleep(0.01)
                cell, drug, ic50s = data
                cell = torch.stack(cell, 0).transpose(1, 0) # Note that this is only neede when geometric 
                                                            # Dataloader is used and no collate.
                cell, drug, ic50s = cell.to(self.device), drug.to(self.device), ic50s.to(self.device)

                self.optimizer.zero_grad()

                # Models predictions of the ic50s for a batch of cell-lines and drugs
                preds = self.model(cell.float(), drug).unsqueeze(1)
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
            train_epoch_mae.append(mean_absolute_error(y_true.detach().cpu(), y_pred.detach().cpu()))
            train_epoch_r2.append(r2_score(y_true.detach().cpu(), y_pred.detach().cpu()))
            train_epoch_pcorr.append(pearsonr(y_true.detach().cpu().numpy().flatten(), 
                                              y_pred.detach().cpu().numpy().flatten()))
                     
            mse, rmse, mae, r2, pcorr = self.validate(self.val_loader)
            val_epoch_losses.append(mse)
            val_epoch_rmse.append(rmse)
            val_epoch_mae.append(mae)
            val_epoch_r2.append(r2)
            val_epoch_pcorr.append(pcorr)

            print("=====Epoch ", epoch)
            print(f"Train      | MSE: {train_mse:2.5f}")
            print(f"Validation | MSE: {mse:2.5f}")

        performance_stats = {
            'train': {
                'mse': train_epoch_losses,
                'rmse': train_epoch_rmse,
                'mae': train_epoch_mae,
                'r2': train_epoch_r2,
                'pcorr': train_epoch_pcorr
            },
            'val': {
                'mse': val_epoch_losses,
                'rmse': val_epoch_rmse,
                'mae': val_epoch_mae,
                'r2': val_epoch_r2,
                'pcorr': val_epoch_pcorr
            }            
        }

        return performance_stats           

    def validate(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(loader, desc='Iteration (Val)'):
                sleep(0.01)
                cl, dr, ic50 = data
                cl = torch.stack(cl, 0).transpose(1, 0)

                preds = self.model(cl.float(), dr).unsqueeze(1)
                ic50 = ic50.to(self.device)
                total_loss += self.criterion(preds, ic50.view(-1,1).float())
                # total_loss += F.mse_loss(preds, ic50.view(-1, 1).float(), reduction='sum')
                y_true.append(ic50.view(-1, 1))
                y_pred.append(preds)
        
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        mse = total_loss / len(loader)
        rmse = torch.sqrt(mse)
        mae = mean_absolute_error(y_true.detach().cpu(), y_pred.detach().cpu())
        r2 = r2_score(y_true.detach().cpu(), y_pred.detach().cpu())
        pearson_corr_coef, _ = pearsonr(y_true.detach().numpy().flatten(), 
                                        y_pred.detach().numpy().flatten())

        return mse, rmse, mae, r2, pearson_corr_coef


class TabGraph_v1(torch.nn.Module):
    def __init__(self):
        super(TabGraph_v1, self).__init__()
        # torch.manual_seed(12345)

        self.cell_emb = nn.Sequential(
            # TODO: the 2784 needs to be an arguments! provide it in the main!
            nn.Linear(2784, 516),
            nn.BatchNorm1d(516),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(516, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
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
                nn.Dropout(p=0.1),
                nn.Linear(128, 128),
                nn.ReLU()
            ]
        )

        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )

    def forward(self, cell, drug):
        cell_emb = self.cell_emb(cell)
        drug_emb = self.drug_emb(drug.x.float(), drug.edge_index, drug.batch)
        concat = torch.cat([cell_emb, drug_emb], -1)
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred       