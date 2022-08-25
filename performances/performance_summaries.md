
# Summary Of Model Performances With Model Informations

`v1_NeuralNN.py`: 

- `Model`
```
Information
-----------
    - uses only Gene Expression information as cell line features

Parameters
----------
    num_epochs    = 100
    batch_size    = 20_000
    learning_rate = 0.001  

Architecture
------------
    Cell Branch
    -----------
        nn.Linear(908, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU()   
    Drug Branch
    -----------    
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU()     
    Full Connected Module
    ---------------------
        nn.Linear(2*128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.ReLU()            

Performance
-----------
    TRAIN
        min  : 0.00000809
        mean : 0.00001327
        max  : 0.00003377
    TEST
        min  : 2.66170430
        mean : 4.24001741
        max  : 10.83005047

Conclusions
-----------
    - highly overfitted
```

- `Model2`
```
TODO
```