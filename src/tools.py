import re
import numpy as np

def clean_data(data):
    data = re.sub("<<[^>]+>>", "", data)        # Removes copyright boxes
    data = re.sub("\"", "", data)               # Removes "
    data = re.sub("[\(\[].*?[\)\]]", "", data)  # Remove brackets (), []
    return data

def load_data(data_file, one_hot=True):
    # Open file
    with open(data_file, "r") as f:
        data = f.read()
   
    # Preprocess data to reduce number of characters
    data = clean_data(data)

    chars = list(set(data))                         # Get list of all characters
    chars = [char for char in sorted(chars)]        # Sort characters
    data_size, char_size = len(data), len(chars)    # Compute size of data and number of unique characters

    # Create dictionaries that associate every character with a unique ID
    char2id, id2char = {c:k for k,c in enumerate(chars)}, {k:c for k,c in enumerate(chars)}
    print("Data has %d characters from which %d are unique." % (data_size, char_size))

    # Convert dataset to one-hot encoded vectors
    I = np.eye(len(chars))
    data_one_hot = [I[char2id[char]] for char in data]

    # Split dataset into 95% training, 5% evaluation data
    train_eval_ratio = 0.95  
    idx = int(train_eval_ratio * len(data_one_hot))
    train_data_one_hot = data_one_hot[:idx]
    eval_data_one_hot = data_one_hot[idx:]

    train_data_size = len(train_data_one_hot)
    eval_data_size = len(eval_data_one_hot)

    return train_data_one_hot, eval_data_one_hot, train_data_size, eval_data_size, char_size, char2id, id2char

def get_batch(text_one_hot, data_size, batch_size, seq_length): # random sequence, one-hot, many-to-one
    # Get random positions
    rand_pos = np.random.randint(0, data_size-seq_length, batch_size)
    # Extract one-hot text batch x and y at random positions
    batch_x_one_hot = np.array([text_one_hot[pos:pos+seq_length] for pos in rand_pos])
    batch_y_one_hot = np.array([text_one_hot[pos+seq_length] for pos in rand_pos])
    return batch_x_one_hot, batch_y_one_hot

def get_eval_data(text_one_hot, data_size, seq_length):
    # Extract one-hot text batch x and y at random positions
    batch_x_one_hot = np.array([text_one_hot[pos:pos+seq_length] for pos in range(len(text_one_hot)-seq_length-1)])
    batch_y_one_hot = np.array([text_one_hot[pos+seq_length] for pos in range(len(text_one_hot)-seq_length-1)])
    return batch_x_one_hot, batch_y_one_hot

def load_data_info(data_file):
    # Open file with data
    with open(data_file, "r") as f:
        data = f.read() 
    data = clean_data(data)                         # Preprocess data
    chars = list(set(data))                         # Get list of all characters
    chars = [char for char in sorted(chars)]        # Sort characters

    # Create dictionaries that associate every character with a unique ID
    char2id, id2char = {c:k for k,c in enumerate(chars)}, {k:c for k,c in enumerate(chars)}
    return char2id, id2char

def text2data(string, seq_length, I, char2id):
    string = (seq_length-len(string))*" " + string
    text_one_hot_ = np.array([I[char2id[char]] for char in string])
    return np.expand_dims(text_one_hot_, axis=0)
