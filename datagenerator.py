import numpy as np
import tensorflow as tf

# Generate trainable dataset
'''
Input: image path, class labels
output: 3-D image samples, one-hot coded labels
'''
class ImageDataGenerator():
    def __init__(self, images, labels, batch_size, num_classes, img_format = 'jpg', shuffle = True):
        self.imgpath = images  # image are stored in a file path
        self.labels = labels
        self.size = len(labels)  # number of samples
        self.img_format = img_format    # default input image is .jpg file
        self.batch_size = batch_size
        self.num_class = num_classes

        if shuffle:
           self.shuffle_list()

        # convert image sample and labels into tensor
        self.imgpath = tf.convert_to_tensor(self.imgpath, dtype = tf.string)  # convert a data into tensor
        self.labels = tf.convert_to_tensor(self.labels, dtype = tf.int32)
        data = tf.data.Dataset.from_tensor_slices((self.imgpath, self.labels))   # create dataset object from  slices of data: each data sample is a dictionary [image, label]
        data = data.map(self.parse_function_train) # mapping: Transformation function, receive input to  a mapping function, and output a new dataset
        data = data.batch(batch_size)   # create dataset that splits into several batches
        self.data = data  # the data we needed

    # shuffle the input image sets
    def shuffle_list(self):
        path = self.imgpath
        labels = self.labels
        permutation = np.random.permutation(self.size)  # create a shuffle sequence by length of data
        self.imgpath = []
        self.labels = []
        for i in permutation:
            self.imgpath.append(path[i])
            self.labels.append(labels[i])

    # Convert image path into 3-D bgr image, convert label into one-hot coding
    def parse_function_train(self, filename, label):  # map_func: each elements in dataset viewed as input of parse_function_train, which is [sample, label]
        one_hot = tf.one_hot(label, self.num_class)  # create one-hot encoded output labels
        img_string = tf.read_file(filename)   # read file from path, filename & output data type = Tensor
        if self.img_format == 'jpg':
            img_decode = tf.image.decode_jpeg(img_string, channels = 3) # decode JPG file into uint8 tensor, channel = 3 which is RGB image output
        elif self.img_format == 'png':
            img_decode = tf.image.decode_png(img_string, channels = 3)
        else:
            print("Error! Can't identify the format of image")

        img_resized = tf.image.resize_images(img_decode, [227, 227])  # resize the image for AlexNet input

        return img_resized, one_hot
