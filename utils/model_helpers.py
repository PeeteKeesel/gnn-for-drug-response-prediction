
import time
import torch
import numpy    as np
import torch.nn as nn

from numpy           import vstack
from tqdm            import tqdm
from sklearn.metrics import mean_squared_error


# ------------------------------------- #
# MAKE BATCH SIZE EQUAL FOR ALL BATCHES # 
# INCLUDING THE LAST ONE                #
# ------------------------------------- #
def make_each_batch_same_size(test_loader, dataset):
    """Temporary helper method to make all batches equally sized by adding zeros
    to the last batch which might have a size smaller then the batch size.

    Args:
        test_loader (torch.utils.data.DataLoader):
            Data loader which contains the full test set in batches.
        dataset (List[np.array])
            List of batches.

    Returns:
        (torch.tensor)
            Tensor which the zeros filled for the last batch to equal the batch size.    
    """
    result_tensor = torch.empty((sum(1 for _, _, _, _ in test_loader), test_loader.batch_size))
    for i in range(len(dataset)): 
        if len(dataset[i]) == test_loader.batch_size:
            result_tensor[i] = torch.tensor(dataset[i])
        else: 
            result_tensor[i] = torch.cat([torch.unsqueeze(torch.tensor(dataset[i]), 0), 
                                          torch.zeros(1, test_loader.batch_size-len(dataset[i]))], dim=1)
    return result_tensor


# ---------------------------- #
# TRAINING & TESTING PER EPOCH #
# ---------------------------- #
def train_and_test_model(
    modeling_dataset,
    model, 
    criterion, 
    optimizer, 
    num_epochs,
    device,
    train_loader,
    test_loader    
):  
    start = time.time()
    loss_values = []
    loss_values_test = []
    total_step = len(train_loader)
    it = iter(train_loader)

    for epoch in range(num_epochs): 
        running_loss_train = 0.0
        model.train()
        # TODO: Pytorch lightening.
        for i, (X_batch, X_batch_cell, X_batch_drug, y_batch) in tqdm(enumerate(train_loader)):
            
            # Set gradients to zero before starting backprop.
            optimizer.zero_grad()

            # X_batch_cell, X_batch_drug
            # X_cell = X_batch_cell.to(device)
            # X_drug = X_batch_drug.to(device)
            X = X_batch.to(device)
            X_cell = X_batch_cell.to(device)
            X_drug = X_batch_drug.to(device)            
            y = y_batch.to(device)

            y_preds = model(X_cell.reshape(X_cell.shape[0], 1, X_cell.shape[1]), 
                            X_drug.reshape(X_drug.shape[0], 1, X_drug.shape[1]))
            loss = criterion(y_preds, y)

            #print(f"Loss: {loss}")

            running_loss_train += loss.item()

            # print(f"Running Loss: {running_loss_train}")

            # Backward and optimize.
            loss.backward()
            optimizer.step()

            #print(f"Optimized! i : {i}")

            if i % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'\
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))             
                break
            
        loss_values.append(running_loss_train / len(modeling_dataset))      
        print(f"loss_values : {np.mean(loss_values):2.4f}")

        # ------------------ #
        # EVALUATE THE MODEL #
        # ------------------ #
        model.eval()
        mse_test = test_model(model, test_loader, device)
        print(f"MSE (test) : {mse_test}")
        loss_values_test += [mse_test]

    return model, loss_values, loss_values_test  

# -------------- #
# Test the model #
# -------------- #
def test_model(model, test_loader, device):
    """Evaluates the model on a test set. 

    Args:
        model (nn.Module):
            The pytorch model to use for evaluation.
        test_loader (torch.utils.data.DataLoader):
            Data loader which contains the full test set in batches.
        device  (torch.device):
            Device on which to save the datasets.

    Returns:    
        (float)
            Mean squared error of the total test set.
    """
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0

        inputs, predictions, actuals = list(), list(), list()
        for i, (X_batch, X_batch_cell, X_batch_drug, y_batch) in tqdm(enumerate(test_loader)):
            X = X_batch.to(device)
            X_cell = X_batch_cell.to(device)
            X_drug = X_batch_drug.to(device)            
            y = y_batch.to(device)

            outputs = model(X_cell.reshape(X_cell.shape[0], 1, X_cell.shape[1]), 
                            X_drug.reshape(X_drug.shape[0], 1, X_drug.shape[1]))
            predicted = outputs.data

            y_preds  = outputs.detach().numpy()
            y_actual = y.detach().numpy() 

            assert y_preds.shape == y_actual.shape,\
                f"y_preds.shape = {y_preds.shape} != {y_actual.shape} = y_actual.shape"
            
            inputs.append(X.detach().numpy())
            predictions.append(y_preds)
            actuals.append(y_actual)

            total += y.size(0)
            # print(f"predicted y : {predicted}    actual y : {y}\n\n\n\n")
            # print(f"actual    y : {y}\n\n")
            correct += np.abs(predicted - y).sum()

        # print(total)
        # print(correct)
        # print(f"Mean absolute difference of the network on the {len(test_loader.dataset.indices)} test values: {correct / total:2.6f}")

        inputs = vstack(inputs)
        #predictions = vstack(predictions)
        #actuals =  vstack(actuals)
        # calculate mse
        loss = nn.MSELoss()

        # Make all the same shape.
        # TODO: Now I have added zeros to the last batch which has a size < 20_000 (=batch size). Change this cause now this is seen as perfect predicitons.    
        actuals_tensor = make_each_batch_same_size(test_loader, actuals)
        predictions_tensor = make_each_batch_same_size(test_loader, predictions)

        mse = loss(actuals_tensor, predictions_tensor)
        return mse
                