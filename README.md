# LittleShakespeare

This is a simple character-level language model that can be trained on arbitrary texts. Here, works by Shakespeare are used. The model consists of a Recurrent Neural Network that is implemented in Python with Tensorflow. This implementation allows to easily change the networks architecture and which RNN cell to use. Different RNN cells such as Basic-, GRU- or LSTM-Cells can be used and compared with each other. Also depth and width of the network can be freely chosen.

This implementation uses a many-to-one approach. This means that the model is trained to predict the next character in a sequence of characters:

```
                                                                                            g
                                                                                            O
                                                                                            |  
          O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
          | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
          O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
          T h e   q u i c k   b r o w n   f o x   j u m p s   o v e r   t h e   l a z y   d o
```
                                  
The example above shows the case where a sequence of characters of length 42 (`seq_length`) is used to predict the 43th character in the sequence. All characters are one-hot encoded. In a dataset that consists only of the letters `a`, `b` and `c` the letters are represented as the vectors `[1,0,0]`, `[0,1,0]` and `[0,0,1]`.

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

The line `num_units = [64, 64]` defines the network's architecture. The number of elements in this list defines the networks depth or how many RNN-cells are to be connected in series. The numbers itself define the cell's hidden layer size.

# Results

The results below show the training and evaluation loss and accuracy for a GRU (orange) and a LSTM (blue) network.

## Training

<div align="center">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/training_loss.svg" width="700">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/training_accuracy.svg" width="700">
</div>

## Evaluation

<div align="center">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/eval_loss.svg" width="700">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/eval_accuracy.svg" width="700">
</div>
