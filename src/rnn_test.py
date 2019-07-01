import re
import os
import numpy as np
import tensorflow as tf

from tools import *

# Paths 
home = os.path.expanduser("~")
logs_dir = home + "/logs/" 
data_dir = home + "/data/"
ckpt_dir = logs_dir + "model.ckpt" 
data_file = data_dir + "shakespeare_1.txt"

# Load data info network was trained on
char2id, id2char = load_data_info(data_file)
I = np.eye(len(char2id))

seq_length = 64

num_letters = 3000                  # Number of characters to be generated
start_sequence = "Even so it was"   # Must be smaller or equal than seq_length
data = text2data(start_sequence, seq_length, I, char2id)

# Start training
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph(ckpt_dir + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint(logs_dir))
    
    X = tf.get_default_graph().get_tensor_by_name("input:0")
    Y = tf.get_default_graph().get_tensor_by_name("output:0")
  
    pred_text = []
    for k in range(num_letters):
        # Print progress
        #print("%.1f" % (100.*(k+1.)/num_letters) + " %", end="\r")
        print("{:.1f} %".format((100.*(k+1.)/num_letters)), end="\r")

        # Predict next letter
        next_letter = sess.run(Y, feed_dict={X:data})

        # Convert output to character id and save id
        next_letter_id = np.argmax(next_letter)
        next_letter_one_hot = I[next_letter_id]
        pred_text.append(id2char[next_letter_id])

        # Add "next_letter_one_hot" to "data", remove first character in sequence
        data[0][:-1,:] = data[0][1:,:]
        data[0][-1,:] = next_letter_one_hot

    # Print results
    print("".join(pred_text))

