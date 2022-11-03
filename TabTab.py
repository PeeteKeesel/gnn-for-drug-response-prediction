import torch
import torch.nn as nn


class TabTabModel(nn.Module):
    def __init__(self):
        super(TabTabModel, self).__init__()
        self.cell_branch = nn.Sequential(
            nn.Linear(3432, 516),
            nn.ReLU(),
            nn.Linear(516, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()          
        )

        self.drug_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()          
        )

        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )     

    def forward(self, cell, drug):
        cell_emb = self.cell_branch(cell)  # Create cell gene vector embedding.
        drug_emb = self.drug_branch(drug)  # Create compound vector embedding.

        concat = torch.cat([cell_emb, drug_emb], 1)
        print(concat)
        x_dim_batch, y_dim_branch, z_dim_features = concat.shape[0], concat.shape[1], concat.shape[2]
        concat = torch.reshape(concat, (x_dim_batch, y_dim_branch*z_dim_features))
        
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred