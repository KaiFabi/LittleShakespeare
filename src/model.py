import tensorflow as tf

def network(x, num_units, num_classes, cell_type):
    x = tf.unstack(x, axis=1) # creates "seq_length" tensors of size (batch_size, num_classes)
    if cell_type == "lstm":
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(num_units=size) for size in num_units]
    elif cell_type == "gru":
        rnn_layers = [tf.nn.rnn_cell.GRUCell(num_units=size) for size in num_units]
    elif cell_type == "basic":
        rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=size) for size in num_units]
    else:
        raise "Error. No such cell type."

    cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32) # Get output of RNN

    return tf.layers.dense(outputs[-1], num_classes, activation=None, use_bias=True)
