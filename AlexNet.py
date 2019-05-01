import tensorflow as tf
import numpy as np

def alexnet(img, keep_prob, num_class):
    # Convolution layer 1
    c1_w = tf.Variable(tf.truncated_normal(shape = [11, 11, 3, 96], stddev = 0.1), name = 'c1_w')
    c1_b = tf.Variable(tf.zeros(96))
    c1_out = tf.nn.conv2d(img, c1_w, strides = [1, 4, 4, 1], padding= 'SAME') + c1_b
    c1_out = tf.nn.relu(c1_out)

    # Max pooling layer 1
    p1 = tf.nn.max_pool(c1_out,
                        ksize = [1,3,3,1],
                        strides = [1,2,2,1],
                        padding = 'VALID'
                        )
    # padding
    '''p1_out = tf.pad(p1, paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), mode = 'CONSTANT')  # padding to let image tensor remain same size, preserve more border information'''
    p1_groups = tf.split(value = p1, num_or_size_splits = 2, axis = 3) # split the data into 2 channels for 2 GPU, each channel size 5*5*48

    # Convolution layer 2
    c2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 48, 256]))
    c2_b = tf.Variable(tf.ones(256))
    # split the output of C1 layer, into 2 partition to GPU
    kernel2_groups = tf.split(value = c2_w, num_or_size_splits = 2, axis = 3)  # split the kernel by channel axis
    bias2_groups = tf.split(value = c2_b, num_or_size_splits = 2, axis = 0)
    c2_up = tf.nn.conv2d(p1_groups[0], kernel2_groups[0], strides = [1,1,1,1], padding = 'SAME') + bias2_groups[0]
    c2_down = tf.nn.conv2d(p1_groups[1], kernel2_groups[1], strides=[1, 1, 1, 1], padding='SAME') + bias2_groups[1]
    c2_out = tf.nn.relu(tf.concat(axis = 3, values = [c2_up, c2_down]))

    # Max pooling layer 2
    p2 = tf.nn.max_pool(c2_out,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID'
                        )
    # output size: 27*27*256
    # padding: prevent image size reduction
    '''p2_out = tf.pad(p2, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    mode='CONSTANT')  # padding to let image tensor remain same size, preserve more border information'''

    # Convolution layer 3: In layer 3, no need to split 2 sections, full connected layer
    c3_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 384]))
    c3_b = tf.Variable(tf.zeros(384))
    c3_out = tf.nn.conv2d(p2, c3_w, strides=[1, 1, 1, 1], padding='SAME') + c3_b
    c3_out = tf.nn.relu(c3_out)

    # padding = 1
    '''p3_out = tf.pad(c3_out, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    mode='CONSTANT')  # padding to let image tensor remain same size, preserve more border information'''

    # Convolution layer 4: split the image tensor into 2 section again
    c4_groups = tf.split(axis = 3, num_or_size_splits = 2, value = c3_out)
    c4_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 384]))
    c4_b = tf.Variable(tf.ones(384))
    kernel4_groups = tf.split(axis = 3, num_or_size_splits = 2, value = c4_w)
    bias4_groups = tf.split(axis = 0, value = c4_b, num_or_size_splits = 2)
    c4_up = tf.nn.conv2d(c4_groups[0], kernel4_groups[0], strides=[1, 1, 1, 1], padding='SAME') + bias4_groups[0] # padding = SAME: convolution don't exceed the edge
    c4_down = tf.nn.conv2d(c4_groups[1], kernel4_groups[1], strides=[1, 1, 1, 1], padding='SAME') + bias4_groups[1]
    c4_out = tf.nn.relu(tf.concat(axis = 3, values = [c4_up, c4_down]))

    # padding
    '''p4_out = tf.pad(c4_out, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    mode='CONSTANT')  # padding to let image tensor remain same size, preserve more border information'''

    # Convolution layer 5: final convolution layer, with a max pooling layer
    c5_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 256]))
    c5_b = tf.Variable(tf.ones(256))
    c5_groups = tf.split(axis = 3, num_or_size_splits = 2, value = c4_out)
    kernel5_groups = tf.split(axis = 3, num_or_size_splits = 2, value = c5_w)
    bias5_groups = tf.split(axis = 0, num_or_size_splits = 2, value = c5_b)
    c5_up = tf.nn.conv2d(c5_groups[0], kernel5_groups[0], strides=[1, 1, 1, 1], padding='SAME') + bias5_groups[0]
    c5_down = tf.nn.conv2d(c5_groups[1], kernel5_groups[1], strides=[1, 1, 1, 1], padding='SAME') + bias5_groups[1]
    c5_out = tf.nn.relu(tf.concat(axis = 3, values = [c5_up, c5_down]))

    # Max pooling layer 5
    p5 = tf.nn.max_pool(c5_out,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID'
                        )

    # flatten output to fully connected layer
    flatten5 = tf.reshape(p5, shape = [-1, 6*6*256]) # -1 means that the size of that dimension is the value that makes the total size remain constant

    # fully connected layer 6: with dropout layer
    w6 = tf.Variable(tf.truncated_normal(shape = [6*6*256, 4096]))
    b6 = tf.Variable(tf.zeros(4096))
    f6_out = tf.matmul(flatten5, w6) + b6
    f6_out = tf.nn.relu(f6_out)

    # dropout layer 6
    dropout6 = tf.nn.dropout(f6_out, keep_prob)

    # fully connected layer 7: followed by dropout layer
    w7 = tf.Variable(tf.truncated_normal(shape=[4096, 4096]))
    b7 = tf.Variable(tf.zeros(4096))
    f7_out = tf.matmul(dropout6, w7) + b7
    f7_out = tf.nn.relu(f7_out)

    # dropout layer 7
    dropout7 = tf.nn.dropout(f7_out, keep_prob)

    # output layer
    w_out = tf.Variable(tf.truncated_normal(shape = [4096, num_class]))
    b_out = tf.Variable(tf.zeros(num_class))
    fc8 = tf.matmul(dropout7, w_out) + b_out

    return fc8

















