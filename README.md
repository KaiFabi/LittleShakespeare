# LittleShakespeare

This is a simple character-level language model that can be trained on arbitrary texts. Here, works by Shakespeare are used. The model consists of a Recurrent Neural Network that is implemented in Python with Tensorflow.

This implementation uses a many-to-one approach. This means that the model is trained to predict the next character in a sequence of characters.

Different RNN cells such as Basic-, GRU- or LSTM-Cells can be used and compared with each other. The implementation is very flexible and both depth and width can be freely chosen.

The code consists of a training and a test module. The test module can be used to create new Shakespeare plays.

Before training, the following parameters can be adjusted: 

```python
# Training Parameters
batch_size = 64
training_steps = 9999999 
eta = 1e-4 # learning rate

# Network Parameters
cell_type = "lstm"      # "lstm", "gru", "basic"
seq_length = 96         # length of input sequence / number of characters / or "time steps"
num_input = char_size   # input vector size / number of unique characters
num_units = [64, 64]    # size of hidden layer
num_classes = char_size # one-hot encoded charakter 
````

The line `num_units = [64, 64]` defines the network's architecture. The number of elements in this list define the networks depth or how many RNN-cells are to be connected in series. The numbers itself define the cell's hidden layer size.

Here are some results:
