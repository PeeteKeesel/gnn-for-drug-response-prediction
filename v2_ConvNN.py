import torch
import torch.nn as nn

# ------------------------------------------------------------------------- #
# In this file we are going to try out using convolutions layers instead of 
# fully connected ones.
# We will first replace the cell branch with a CNN.
# ------------------------------------------------------------------------- #

class ModelConv1(nn.Module):
    def __init__(self):
        super(ModelConv1, self).__init__()
        # print("\nCALLED: def __init__(self)")
        # print(100*"-")
        # --------------- #
        # Cell CNN branch #
        # --------------- #
        self.cell_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=1),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
            #nn.Conv1d(in_channels=1, out_channels=50, kernel_size=1, stride=1), # in_channels=1, since we flattened. If not flattened it could be 3 (features).
            # nn.Tanh(), # other option, from DeepCPR paper.
            #nn.MaxPool1d(kernel_size=3, stride=3),
            # nn.Dropout(p=0.1),
            #nn.Conv1d(in_channels=50, out_channels=80, kernel_size=7, stride=1),
            # nn.BatchNorm1d(80),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=1, stride=1),
            # nn.Linear(in_features=294, out_features=128)
            #nn.ReLU()
        )

        self.cell_branch_fcn = nn.Sequential(
            # nn.Linear(16*897, 256),
            nn.Linear(4768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # print(100*'-')
        # print("self.cell_branch")
        # print(self.cell_branch)
        # print(100*"-")

        # --------------- #
        # Drug CNN branch #
        # --------------- #
        self.drug_branch = nn.Sequential(
            nn.Linear(256, 128),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()          
            #nn.Conv1d(in            
            # BEFORE
            # nn.Conv1d(in_channels=1, out_channels=40, kernel_size=7, stride=1), # in_channels=1, since we flattened. If not flattened it could be 3 (features).
            # nn.BatchNorm1d(40),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=3),
            # # nn.Dropout(p=0.1),
            # nn.Conv1d(in_channels=40, out_channels=20, kernel_size=7, stride=1),
            # nn.BatchNorm1d(20),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=1, stride=1),
            # nn.Linear(in_features=77, out_features=128)
        )
        # print("self.drug_branch")
        # print(self.drug_branch)
        # print(100*"-")

        # --- #
        # FCN #
        # --- #
        # Feed the concatenated vector into a Fully-Connected-Network.
        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        # print("\nself.fcn")
        # print(self.fcn)
        # print(100*"-")        

    def forward(self, cell, drug):
        # print("\nCALLED: forward(self, cell, drug)")
        # Create cell gene vector.
        # print(torch.isnan(cell).sum())
        # print(torch.isinf(cell).sum())

        # print(f"\nCell line INPUT shape: {cell.shape}")
        out_cell = self.cell_branch(cell)
        # print(f"Cell line OUTPUT shape: {out_cell.shape}\n")
        out_cell = torch.flatten(out_cell, start_dim=1, end_dim=2)
        # print(f"Cell line OUTPUT shape (after flattening): {out_cell.shape}\n")
        out_cell = self.cell_branch_fcn(out_cell)
        # print(f"Cell line OUTPUT shape (after FC): {out_cell.shape}\n")
        out_cell = torch.reshape(out_cell, (out_cell.shape[0], 1, out_cell.shape[1]))
        # print(f"Cell line OUTPUT shape (after reshaping): {out_cell.shape}\n")

        # Create compound vector.
        # print(f"\nDrug INPUT shape: {drug.shape}")
        # print(drug)
        # print(torch.isnan(drug).sum())
        # print(torch.isinf(drug).sum())
        compound_vector = self.drug_branch(drug)
        out_drug = compound_vector
        # print(f"Drug OUTPUT shape: {drug.shape}\n")

        # print(f"\n\nSUMMARY\n{100*'+'}")
        # print(f"     out_cell.shape: {out_cell.shape}")
        # print(f"     out_drug.shape: {out_drug.shape}")
        # ----------------------------------------------------- #
        # Concatenate the outputs of the cell and drug branches #
        # ----------------------------------------------------- #
        #concat = torch.concat([out_cell, out_drug], dim=1)
        concat = torch.cat([out_cell, out_drug], 1)
        x_dim_batch, y_dim_branch, z_dim_features = concat.shape[0], concat.shape[1], concat.shape[2]
        # print(f"Before reshaping --> concat.shape: {concat.shape}")
        concat = torch.reshape(concat, (x_dim_batch, y_dim_branch*z_dim_features))
        # print(f"After reshaping --> concat.shape: {concat.shape}")
        # Create vertical vector.
        # concat = torch.unsqueeze(concat, 2)
        # print(f"After unsqueezing --> concat.shape: {concat.shape}")
        
        # ------------------------------- #
        # Run the Fully Connected Network #
        # ------------------------------- #
        y_pred = self.fcn(concat)
        # print(f"Shape of y_pred : {y_pred.shape}")
        # print(y_pred)
        y_pred = y_pred.reshape(y_pred.shape[0])
        # print(f"Shape of y_pred : {y_pred.shape}")
        return y_pred
 