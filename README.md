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

**Loss**
<div align="center">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/training_loss.svg" width="700">
</div>

**Accuracy**
<div align="center">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/training_accuracy.svg" width="700">
</div>

## Evaluation

**Loss**
<div align="center">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/eval_loss.svg" width="700">
</div>

**Accuracy**
<div align="center">
<img src="https://github.com/KaiFabi/LittleShakespeare/blob/master/results/eval_accuracy.svg" width="700">
</div>

In this example, the GRU network performed slightly better compared to the LSTM network. However, both networks tend to overfit. This may be due to the fact that only a small part of Shakespeare's works was used for training (no GPU available). The over-fitting problem could be reduced with more data, for example.

# Text generation

Here is an example text with 3000 characters that the trained network produced:

```
he breath I with my self alter thee she made.


                     13
  Why should I comfort see the worth's stard
  The love, than receivers that which thou willled,
  And they shall steepent of than thy self:
     But die rexill thee,
  The illed I do wit with are spor still,
  To seem than thou wilt so prown thee say seeen.


                     147
  For they with love in your self thy fair thee,
  And true, wherein hath my love should in thee,
    To shall have though my love still of befriel,
    The love, and then my self I lovers'  
  Then did my love still with the heart,
  That thou art of this will be deligel,
  Nor that which in thy self I dracfors,
  Then receive thee my love to thee to thee 
      Yeur,  
  There agn exall the world with all to be.


                     114
  That thou art of that where thou art unuse,
  And the withonelen with all my self still,
    Wherein hate soul, nor me with my sind,
  And straight the will or sick me was I bake.
    But thou art of thy self I look no bid,
  As his sweath, but heaven to show thou sheather shorn
    But from the world me that forbs of his green wit.


                     103
  Alt yetence strong the worttes a beared,
  Then in the stars of many, not be so slowers,
    And thence that I am not the world's fair,
    To commil that thou dost recoulthly shad
  The wastes beauty is a besad,
  That do not so sun no not they dear,
  The leares or all thy self a wild do fortumeat:
  So ere then with me, thy worth the world must show,
    The love, and thou with our love to thee.
    But from your part, though thy shade lose strange.


                     14 
  Whend I aution mad'sthing sorn,
  And then liker thee dearier of my love stand,
  And look anen the have sounds of thee conseets:
    If so, beauteous and shall stand the sun.

                     'e,  
  Lett still surming not the world-lie friend.
  Now in the breath I am a true my self,
  And love in it seeming of your day,
  The ractly days are so short,
  Comounded their sweet lose selfoun thou decease,
  So that for thee, not forguhard and see dead,
  And they seen with the hearth my love to thee.
    To commil the light, which thou wilt still.


                      113
  So my love should in thee I sorn with make,
  And the beauty of thy self a doty new,
  The subseres nothing all thy strong died,
  And beauty shall love to the world's,
  That they most of sicher face so short,
  When I no nor beauty still with thee, and cure,
  That you of your parts of the world's nothing sweet
  To make such in my love's broods a dead,
  And to the love, and then my self uprew?
  The elst thou wilt is but a pained part,
  And the beauty shall beauty stores be,
  For thou art so slay the world in store,
  And present of this will be thine eyes be.
  No it the stars that with my self I'll deeds,
  Will in their ranking the store, or breathesh.
    These branmoned with most in the world will,
  And beauty's fue lose breather
```
