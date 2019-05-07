# Auto-Encoder model

## Running instructions:

Executing

`>> python autoencoder.py`

will build, train and save a simple fully connected autoencoder model 
with default parameters specified in the FLAGS in `autoencoder.py`. 
A folder will be created in the current directory. In the folder checkpoints
of the model will be saved at different times during training. The successive batch
and epoch losses will be saved in a pickle file. Finally, some printout will be written
during training onto a text file. 

To specify any hyperparameters, one can add options to the command as follows:

`>> python autoencoder.py --num_epochs 30 --num_layers 4 --num_bottleneck_units 3 --train_dir '4layers_AE' `

This will build a four layered autoencoder (4 layers in the encoder and 4 layers in the decoder) with the 
default number of units, a bottleneck of size 3 and it will train it for 30 epochs. 
The result will be saved in folder named '4layers_AE'.


### Using bash

The file 'run_AEs.sh' is an example of how we can use bash to automate the running of many models with 
varying hyperparameter values. To execute it, simply run

`>> ./run_AEs.sh`


### Some remarks
1) The current code uses as input RANDOM data. This has to be changed within the code.
2) Currently the model assumes that the data distribution is between 0 and 1 (either make sure
your data is normalised to the unit norm or change the activation function of the last layer in the decoder).
3) There are MANY things that can (and should) be considered to improve the model architecture and/or the 
training. To name a few: adding noise to the inputs (denoising AE), regurlarisation (sparse AE),
type of activation function (sigmoid VS reLu VS ...).
