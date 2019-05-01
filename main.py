import os
import numpy as np
import tensorflow as tf
import AlexNet as Alexnet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
import matplotlib.pyplot as plt

def main():

    # Define hyperparameters
    learning_rate = 1e-4
    num_epochs =5
    train_batch_size = 100
    test_batch_size = 100
    dropout_rate = 0.5
    num_classes = 2  # number of classes
    display_step = 2  # 經過多少step後，計算accuracy並顯示出來

    filewriter_path = "./tmp/tensorboard"  # 存储tensorboard文件
    checkpoint_path = "./tmp/checkpoints"  # 训练好的模型和参数存放目录

    image_format = 'jpg'  # data type
    file_name_of_class = ['cat',
                          'dog']  # cat对应标签0,dog对应标签1。默認圖片包含獨立類別，可用此來建立label 列表

    # Dataset from file directory
    train_dataset_paths1 = ['D:/Codepractice/Dataset/dogs-vs-cats-redux-kernels-edition/train/cat/*.jpg',
                           'D:/Codepractice/Dataset/dogs-vs-cats-redux-kernels-edition/train/dog/*.jpg']

    # Dataset from file directory: floydhub dataset: name 'my_data'
    train_dataset_paths2 = ['/my_data/cat/*.jpg',
                           '/my_data/dog/*.jpg']  # train dataset path directory
    test_dataset_paths = []  # test data file directory


    # create a lists for all image path
    train_image_paths = []
    train_labels = []
    # Read in all data, create list of all images
    for train_dataset in train_dataset_paths2:
        img_train = glob.glob(train_dataset)  # read all files that match JPG in the directory
        train_image_paths.extend(img_train)   # Cat & Dog image training data list

    for image_path in train_image_paths:  # Create training data labels
        image_file_name = image_path.split('/')[-1]  # cut the path name and preserve final short path name, which contains discription of image
        for i in range(num_classes):
            if file_name_of_class[i] in image_file_name:  # if class name contains in the image path name (contains the image description)
                train_labels.append(i)
                break

    # Create the Dataset object that imports training data
    train_data = ImageDataGenerator(
        images=train_image_paths,
        labels=train_labels,
        batch_size=train_batch_size,
        num_classes=num_classes,
        img_format=image_format,
        shuffle=True)

    # get Iterators
    '''
    Iterator: 設定如何run dataset的迭代
    含有兩個method:
    iterator.initialize: 將迭代重新初始化
    iterator.get_next(): 迭代到下一個sample
    '''
    # Iterator define the method we pick up element(batches of sample) from the dataset
    # Create a initializable iterator, that we could feed x ltater
    train_iterator = train_data.data.make_initializable_iterator()
    # further defined
    training_initalizer = train_iterator.initializer
    # Define next batch (element) for Iterator to get
    train_next_batch = train_iterator.get_next()

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # alexnet
    fc8 = Alexnet.alexnet(x, keep_prob, num_classes)

    # Using name scope to name the variables, so we can call these variable to show in the graph
    # loss
    with tf.name_scope('loss'):
      loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8,
                                                                            labels=y))
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)

    # accuracy
    correct_pred = tf.equal(tf.argmax(fc8, 1), tf.argmax(y, 1)) # tf.argmax return the max probability(class) index of output
    # tf.equal return True, False table for the dataset
    # tf.cast convert the True, False into 1, 0 for dataset: [0, 0, 1, 0, 1, .....]
    # tf.reduce_mean compute mean value of dataset, correct prediction is 1, thus mean value is prediction accuracy of the dataset
    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # reduce_mean(tensor, axis, keep_dim): compute mean value of tensor along given axis. If no axis is given, compute mean of all values in tensor.

    # Tensorboard: to show network module and training process, save module and data as ''Event" , and output to port to connect to tensorboard
    # To observe loss and accuracy along training epoch, save two scalar for summary to graph
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()  # merge all summary in graph
    writer = tf.summary.FileWriter(filewriter_path) # write all summary into hardware, create an event file in path to store the summary

    # saver: save parameters for further use
    saver = tf.train.Saver() # to save the variables after trained, and restore variables to analyze

    # Define number of batches
    train_batches_per_epoch = int(np.floor(train_data.size / train_batch_size))
    #test_batches_per_epoch = int(np.floor(test_data.data_size / test_batch_size))

    # Start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./tmp/checkpoints/model_epoch18.ckpt")
        # Tensorboard
        writer.add_graph(sess.graph)  # add new session graph to current event file

        print("{}: Start training...".format(datetime.now()))
        print("{}: Open Tensorboard at --logdir {}".format(datetime.now(),
                                                           filewriter_path))
        for epoch in range(num_epochs):
            sess.run(training_initalizer)  # reset iterator, get batches of train data one by one batch
            #print(sess.run(train_next_batch[0]))
            print("{}: Epoch number: {} start".format(datetime.now(), epoch + 1)) # show the clock time of each training epoch start

            # train
            train_acc = 0
            train_count = 0
            for step in range(train_batches_per_epoch):  # for each batch of samples
                img_batch, label_batch = sess.run(train_next_batch)
                loss, _= sess.run([loss_op, train_op], feed_dict={x: img_batch,
                                                                   y: label_batch,
                                                                   keep_prob: dropout_rate})
                if step % display_step == 0:
                    print("{}: loss = {}".format(datetime.now(), loss))
                    s = sess.run(merged_summary, feed_dict={x: img_batch,  # save to graph the current loss & accuracy
                                                            y: label_batch,
                                                            keep_prob: 1.})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)  # add  new summary to current event
                    # global step value: record the global total steps for the summary
                # Show trainin accuracy
                acc = sess.run(accuracy, feed_dict={x: img_batch, y:label_batch, keep_prob: 1.0})
                train_acc += acc
                train_count += 1

            train_acc /= train_count
            print('{}: Training Accuracy: = {:.4f}'.format(datetime.now(), train_acc))

            # save model
            print("{}: Saving checkpoint of model...".format(datetime.now()))
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            # this epoch is over
            print("{}: Epoch number: {} end".format(datetime.now(), epoch + 1))


if __name__ == '__main__':
    main()

