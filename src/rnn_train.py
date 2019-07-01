import os
import re
import numpy as np
import tensorflow as tf

from tools import *
from model import network

# Paths 
home = os.path.expanduser("~")
logs_dir = home + "/logs/" 
data_dir = home + "/data/"
ckpt_dir = logs_dir + "model.ckpt" 
data_file = data_dir + "shakespeare_1.txt"

# Load data
train_data_one_hot, eval_data_one_hot, train_data_size, eval_data_size, char_size, char2id, id2char = load_data(data_file)

# Training Parameters
batch_size = 16
training_steps = 9999999 
eta = 1e-4 # learning rate

# Network Parameters
cell_type = "gru"           # "lstm", "gru", "basic"
seq_length = 64             # length of input sequence / number of characters / or "time steps"
num_input = char_size       # input vector size / number of unique characters
num_units = [64, 64]        # size of hidden layer # 64
num_classes = char_size     # one-hot encoded charakter 

disp_train_every_n_steps = 1000
disp_eval_every_n_steps = 10000
save_model_every_n_steps = 10000

# Placeholder
X = tf.placeholder("float", [None, seq_length, num_input], name="input")
Y = tf.placeholder("float", [None, num_classes], name="label")
learning_rate = tf.placeholder("float", name="learning_rate")

logits = tf.identity(network(X, num_units, num_classes, cell_type=cell_type), name="output")

# Compute loss 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model 
pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

# Tensorboard Summary
# Training 
summary_learning_rate = tf.summary.scalar(name="learning_rate", tensor=learning_rate)
summary_training_loss = tf.summary.scalar(name="training_loss", tensor=loss)
summary_training_accuracy = tf.summary.scalar(name="training_accuracy", tensor=accuracy)
summary_train = tf.summary.merge([summary_learning_rate, summary_training_loss, summary_training_accuracy])
# Evaluation 
summary_eval_loss = tf.summary.scalar(name="eval_loss", tensor=loss)
summary_eval_accuracy = tf.summary.scalar(name="eval_accuracy", tensor=accuracy)
summary_eval = tf.summary.merge([summary_eval_loss, summary_eval_accuracy])

# Get evaluation data
eval_x, eval_y = get_eval_data(eval_data_one_hot, eval_data_size, seq_length)

# Start training
with tf.Session() as sess:
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    train_writer = tf.summary.FileWriter(logs_dir, sess.graph) # TB

    for step in range(training_steps):

        # Get training data
        batch_x, batch_y = get_batch(train_data_one_hot, train_data_size, batch_size, seq_length)

        # Train model
        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y, learning_rate:eta})

        if step % disp_train_every_n_steps == 0: # Calculate batch loss and accuracy for training
            summary = sess.run(summary_train, feed_dict={X: batch_x, Y: batch_y, learning_rate:eta})
            train_writer.add_summary(summary, step) #TB

        if step % disp_eval_every_n_steps == 0: # Calculate batch loss and accuracy for evaluation
            summary = sess.run(summary_eval, feed_dict={X: eval_x, Y: eval_y})
            train_writer.add_summary(summary, step) #TB

        if step % save_model_every_n_steps == 0: # Save progress of model
            saver.save(sess, ckpt_dir)
