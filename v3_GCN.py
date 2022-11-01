import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool, global_max_pool


class GraphTab_v1(torch.nn.Module):
    def __init__(self):
        super(GraphTab_v1, self).__init__()
        torch.manual_seed(12345)

        self.dropout_p = 0.1

        # Drug branch.
        self.drug_nn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # TODO: nn.Dropout(self.dropout_p),
            nn.Linear(128, 128),
            nn.ReLU()     
            # TODO: nn.Dropout(self.dropout_p),       
        )
        print(f"self.drug_nn: {self.drug_nn}")

        # Cell-line graph branch. Obtains node embeddings.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.sequential.Sequential
        self.cell_emb = Sequential('x, edge_index', 
            [
                (GCNConv(in_channels=4, out_channels=256), 'x, edge_index -> x1'), # TODO: try GATConv() vs GCNConv()
                nn.ReLU(inplace=True),
                ## nn.BatchNorm1d(num_features=128),
                ## nn.Dropout(self.dropout_p),
                (GCNConv(in_channels=256, out_channels=256), 'x1, edge_index -> x2'),
                nn.ReLU(inplace=True),
                (global_mean_pool, 'x2, batch -> x3'), 
                # Start embedding
                nn.Linear(256, 128),
                nn.ReLU(),
                ## nn.Dropout(self.dropout_p),
                nn.Linear(128, 128),
                nn.ReLU()
                ## nn.Dropout(self.dropout_p)            
            ]
        )
        print(f"self.cell_emb: {self.cell_emb}")

        self.cell_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(self.dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Dropout(self.dropout_p)
        )

        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, cell, drug):
        print(drug)
        drug_emb = self.drug_nn(drug)
        print(f"drug_emb.shape: {drug_emb.shape}")
        print(f"drug_emb: {drug_emb}")

        # cell_gnn_out = self.cell_gnn(cell.x, cell.edge_index)
        # print(f"cell_gnn_out: {cell_gnn_out}")
        # # Readout layer.
        # cell_gnn = global_mean_pool(cell_gnn_out, cell) # [batch_size, hidden_channels]
        # cell_emb = self.cell_emb(cell_gnn)
        cell_emb = self.cell_emb(cell.x, cell.edge_index)
        print(f"cell_emb: {cell_emb}")


        # ----------------------------------------------------- #
        # Concatenate the outputs of the cell and drug branches #
        # ----------------------------------------------------- #
        concat = torch.cat([cell_emb, drug_emb], 1)
        x_dim_batch, y_dim_branch, z_dim_features = concat.shape[0], concat.shape[1], concat.shape[2]
        print(f"concat.shape: {concat.shape}")
        concat = torch.reshape(concat, (x_dim_batch, y_dim_branch*z_dim_features))
        
        # ------------------------------- #
        # Run the Fully Connected Network #
        # ------------------------------- #
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred
        
