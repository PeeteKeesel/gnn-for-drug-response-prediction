
# Summary Of Model Performances With Model Informations

<ins>GraphTab:</ins>
```
    Finished reading drug response matrix: (137835, 9)
    Finished reading cell-line graphs: Data(x=[696, 4], edge_index=[2, 7794])
    Finished reading drug SMILES dict: 181
    DRM Number of unique cell-lines: 856
    Finished building GraphTabDataset!
    GraphTabDataset Summary
    =======================
    # observations : 137835
    # cell-lines   : 856
    # drugs        : 181
    # genes        : 696
    Full     shape: (137835, 9)
    train    shape: (110268, 9)
    test_val shape: (27567, 9)
    test     shape: (13783, 9)
    val      shape: (13784, 9)

    train_dataset:
    GraphTabDataset Summary
    =======================
    # observations : 110268
    # cell-lines   : 856
    # drugs        : 181
    # genes        : 696


    test_dataset:
    GraphTabDataset Summary
    =======================
    # observations : 13783
    # cell-lines   : 856
    # drugs        : 181
    # genes        : 696


    val_dataset:
    GraphTabDataset Summary
    =======================
    # observations : 13784
    # cell-lines   : 856
    # drugs        : 181
    # genes        : 696
    Finished creating pytorch training datasets!
    Number of batches per dataset:
      train : 111
      test  : 14
      val   : 14
    device: cpu
    Iteration (Train): 100%|████████████████████████████████████| 111/111 [1:06:59<00:00, 36.21s/it]
    Iteration (Val): 100%|██████████████████████████████████████████| 14/14 [01:55<00:00,  8.26s/it]
    =====Epoch  0
    Train      | MSE: 10.25914
    Validation | MSE: 8.10883
    Iteration (Train): 100%|████████████████████████████████████| 111/111 [1:01:04<00:00, 33.01s/it]
    Iteration (Val): 100%|██████████████████████████████████████████| 14/14 [01:22<00:00,  5.89s/it]
    =====Epoch  1
    Train      | MSE: 7.83486
    Validation | MSE: 12.26988
```