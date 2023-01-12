# numpy-nn

The dataset includes a list of arrays with a combination of 0's and 1's. the neural network performs classification. data is labelled with a 1 if an array has an even number of 1's and 0 otherwise. 
e.g. [0,1,0,1] is labeled with 1
e.g. [0,0,0,0,1] is labeled with 0 


`data_handler.py` has 2 relevent variables
`n_per_entry` determines how many 0/1s make up an array
`n_entries` determines how many total arrays exist

`run_nn.py` has 2 relevent variables
 
`lr` and `epochs` are self explanatory. 
`layers` is the structure of the NN. `layers = [3,4,1]` means the input layer has 3 nodes, a hidden layer has 4 nodes, and the output layer has 1 node.

Example output:
---------- ^Handling data...^ ----------\n
Accuracy: 0.502\n
-------------------- ^ACCURACY (UNTRAINED)^ --------------------
Epoch: 500\n
Epoch: 1000
Epoch: 1500
Epoch: 2000
Epoch: 2500
Epoch: 3000
Epoch: 3500
-------------------- ^TRAINING^ --------------------
Accuracy: 1.0
-------------------- ^ACCURACY (TRAINED)^ --------------------
Accuracy: 0.984
-------------------- ^ACCURACY (TRAINED) (VALIDATION)^ --------------------

