import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


class GCN_v1(torch.nn.Module):
    def __init__(self):
        super(GCN_v1, self).__init__()

        # Drug branch.
        self.drug_branch = nn.Sequential(
            nn.GCNConv()
        )

        # Cell-line branch.
        self.cell_branch = nn.Sequential(

        )
