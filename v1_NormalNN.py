import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # print("\nCALLED: def __init__(self)")
        # print(100*"-")
        # --------------- #
        # Cell CNN branch #
        # --------------- #
        self.cell_branch = nn.Sequential(
            nn.Linear(908, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()          
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
 

# -------------------------------------------------- #
# Goal of this model is to prevent overfitting as in 
# the previous one.
# -------------------------------------------------- #
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # --------------- #
        # Cell CNN branch #
        # --------------- #
        self.cell_branch = nn.Sequential(
            nn.Linear(908, 516),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(516, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()          
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

        # --- #
        # FCN #
        # --- #
        # Feed the concatenated vector into a Fully-Connected-Network.
        self.fcn = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )     

    def forward(self, cell, drug):
        # Create cell gene vector.
        out_cell = self.cell_branch(cell)

        # Create compound vector.
        compound_vector = self.drug_branch(drug)
        out_drug = compound_vector
        # ----------------------------------------------------- #
        # Concatenate the outputs of the cell and drug branches #
        # ----------------------------------------------------- #
        concat = torch.cat([out_cell, out_drug], 1)
        x_dim_batch, y_dim_branch, z_dim_features = concat.shape[0], concat.shape[1], concat.shape[2]
        concat = torch.reshape(concat, (x_dim_batch, y_dim_branch*z_dim_features))
        
        # ------------------------------- #
        # Run the Fully Connected Network #
        # ------------------------------- #
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred

# -------------------------------------------------- #
# This model will use the layer structure of the transcriptomic 
# input (gene expression data) explained in
# https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/36/Supplement_2/10.1093_bioinformatics_btaa822/4/btaa822_supplementary_material.pdf?Expires=1663850267&Signature=vN~p7HuM218~JBcokwxpr5w-2gfj3IslF~qhq-O0GNLtuksjxtxYibdz-DnpbmugYyNc2Klla~olARTV6jlKhwL6Vhmtl66EyQGJuXJHQTiopeJzb15BL0jPGjwnEzMWzr~p7YwT8tF8-chqGPY1E5T3J2i1Wp0t8k~eWdDl5DLpX84r8NqA7T5GYHAwurabYWFIvFvjLYpEwEtyWPt4qJ4UqwyUHmQ56b~MZ0EywmL1cP6OQp30K~cYk2kdGPrIIPileYFFJYlWZ7QMjJjdbkzxgbfoxPE~uCcIPqsGjI2M9kkAlUI9FgkOuDbtlLcjMPL80OLZjHMxDajfiNsx7A__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA
# paper: http://liuqiao.me/files/DeepCDR.pdf
# -------------------------------------------------- #
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        # --------------- #
        # Cell CNN branch #
        # --------------- #
        self.cell_branch = nn.Sequential(
            nn.Linear(908, 256),
            nn.Tanh(),
            nn.BatchNorm1d(1),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU()          
        )

        # --------------- #
        # Drug CNN branch #
        # --------------- #
        self.drug_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU()          
        )

        # --- #
        # FCN #
        # --- #
        # Feed the concatenated vector into a Fully-Connected-Network.
        self.fcn = nn.Sequential(
            nn.Linear(100+100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )     

    def forward(self, cell, drug):
        # Create cell gene vector.
        out_cell = self.cell_branch(cell)

        # Create compound vector.
        compound_vector = self.drug_branch(drug)
        out_drug = compound_vector
        # ----------------------------------------------------- #
        # Concatenate the outputs of the cell and drug branches #
        # ----------------------------------------------------- #
        concat = torch.cat([out_cell, out_drug], 1)
        x_dim_batch, y_dim_branch, z_dim_features = concat.shape[0], concat.shape[1], concat.shape[2]
        concat = torch.reshape(concat, (x_dim_batch, y_dim_branch*z_dim_features))
        
        # ------------------------------- #
        # Run the Fully Connected Network #
        # ------------------------------- #
        y_pred = self.fcn(concat)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return y_pred