import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import shuffle

from PIL import Image

from ai.conv import ImageOperations
from ai.conv.NN_Image import NN_Image

min = 0
max = 0


class NN_Sharpen:
    def __init__(self):
        self.learning_rate = 0.0001
        self.batch_size = 5
        self.input_width = 256
        self.input_height = 256
        self.layer_info = []

    def push_conv(self, name, shape, downsample, activation, in_layer):
        self.layer_info.append(in_layer.get_shape().as_list())
        W = tf.get_variable("W_" + name, initializer=tf.truncated_normal(shape, stddev=0.1))
        B = tf.get_variable("B_" + name, initializer=tf.ones(shape=[shape[-1]]) / 10)
        out_layer = activation(tf.add(tf.nn.conv2d(in_layer, W,
                                                   strides=[1, downsample, downsample, 1],
                                                   padding='SAME'), B), name="CONV_" + name)
        return out_layer

    def pop_conv(self, name, shape, upsample, activation, in_layer):
        conv_shape = self.layer_info.pop()
        W = tf.get_variable("W_" + name, initializer=tf.truncated_normal(shape, stddev=0.1))
        B = tf.get_variable("B_" + name, initializer=tf.ones([shape[-2]]) / 10)
        layer = activation(tf.add(tf.nn.conv2d_transpose(in_layer, W,
                                                         conv_shape,
                                                         strides=[1, upsample, upsample, 1],
                                                         padding='SAME')
                                  , B), name="DECONV_" + name)
        return layer

    def last_layer(self, shape, in_layer, upsample):
        W2 = tf.get_variable("W_out", initializer=tf.truncated_normal(shape, stddev=0.1))
        B2 = tf.get_variable("B_out", initializer=tf.ones([3]) / 10)
        layer = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(in_layer, W2,
                                                         [1, self.input_width, self.input_height, 3],
                                                         strides=[1, upsample, upsample, 1],
                                                         padding='SAME')
                                  , B2), name="Y")
        return layer

    def train_on_images(self, train, validate, num_of_elements):
        start = time.time()
        self.sess = tf.Session()
        # network set-up
        self.img_label = tf.placeholder(tf.float32, shape=[1, self.input_width, self.input_height, 3], name="img_label")
        self.input = tf.placeholder(tf.float32, shape=[1, self.input_width, self.input_height, 3], name="input")

        # f_sizeX, fsizeY, in channels, out channels
        layer = self.push_conv("1", [5, 5, 3, 16], 2, tf.nn.leaky_relu, self.input)
        layer = self.push_conv("2", [3, 3, 16, 32], 2, tf.nn.leaky_relu, layer)
        layer = self.push_conv("3", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.push_conv("4", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)

        layer = self.pop_conv("d1", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.pop_conv("d2", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.pop_conv("d3", [3, 3, 16, 32], 2, tf.nn.leaky_relu, layer)
        self.Y = self.last_layer([5, 5, 3, 16], layer, 2)

        train_cost = tf.losses.mean_squared_error(self.Y, self.img_label)
        optim = tf.train.AdamOptimizer(self.learning_rate).minimize(train_cost)
        # logging
        for value in [train_cost]:
            tf.summary.scalar("train_cost.{}".format(time.time()), value)



        summaries = tf.summary.merge_all()
        log = tf.summary.FileWriter("logs", self.sess.graph)

        # Create a saver object which will save all the variables
        self.saver = tf.train.Saver()

        # initialize
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        indices = list(range(num_of_elements))
        i = 0
        shuffle(indices)
        while True:
            for idx in indices:
                # prepare input and label image

                try:
                    # If stopfile exists then we stop training
                    open(".\\stopfile")
                    print("learning stops - took {}".format(time.time() - start))
                    return
                except IOError:
                    None

                nn_img_train = NN_Image(train[idx])
                nn_img_train.getNumPyArr()
                chunks_train = nn_img_train.get_image_chunks(chunk_size=(self.input_width, self.input_height))
                nn_img_label = NN_Image(validate[idx])
                nn_img_label.getNumPyArr()
                chunks_label = nn_img_train.get_image_chunks(chunk_size=(self.input_width, self.input_height))
                # iterate over chunks
                for x in range(chunks_train.shape[0]):
                    for y in range(chunks_train.shape[1]):
                        chunk_train = chunks_train[x, y]
                        chunk_label = chunks_label[x, y]
                        # reshape images to have a leading 1 dimension
                        img_shape = chunk_train.shape
                        img_train_reshaped = chunk_train.reshape(1, img_shape[0], img_shape[1], 3)
                        img__label_reshaped = chunk_label.reshape(1, img_shape[0], img_shape[1], 3)
                        output_val, loss_val, _, summaries_val = self.sess.run([self.Y, train_cost, optim, summaries],
                                                                               feed_dict={
                                                                                   self.input: img_train_reshaped,
                                                                                   self.img_label: img__label_reshaped
                                                                               })
                        log.add_summary(summaries_val, i)
                        print("iter:{} loss:{}".format(i, loss_val))
                        i += 1

                        if i % 100 == 0:
                            tf.summary.image("given - iter{}".format(str(i)), self.input)
                            tf.summary.image("result - iter.{}".format(str(i)), self.Y)
                            self.sharpen("data\\", "test4", ".jpg", str(i))

    def show_image(self, img):
        # out_resized = img.reshape((img.shape[0], img.shape[1], 3))
        plt.imshow(img[:, :, :])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()



    def sharpen(self, baseDir, imgName, imgExtension, iteration=""):
        nn_img = NN_Image(baseDir + imgName + imgExtension)
        nn_img.getNumPyArr()
        chunks = nn_img.get_image_chunks(chunk_size=(self.input_width, self.input_height))
        x_max = chunks.shape[0] * chunks.shape[2]
        y_max = chunks.shape[1] * chunks.shape[3]
        chunk_x = chunks.shape[2]
        chunk_y = chunks.shape[3]
        output = np.empty(shape=(x_max, y_max, 3))

        for x in range(chunks.shape[0]):
            for y in range(chunks.shape[1]):
                chunk = chunks[x, y]

                x_shape = chunk.shape[0]
                y_shape = chunk.shape[1]

                # reshape image to have a leading 1 dimension
                img_reshaped = chunk.reshape(1, x_shape, y_shape, 3)
                output_val = self.sess.run(self.Y, feed_dict={
                    self.input: img_reshaped,
                    self.img_label: img_reshaped
                })
                # normalize for displaying
                output_val[output_val > 1.0] = 1.0
                # add chunk to final image
                output[x * chunk_x:x * chunk_x + chunk_x, y * chunk_y:y * chunk_y + chunk_y, :] = output_val
        # self.show_image(output)
        ImageOperations.saveFile(output, ".\\{}.jpg".format(baseDir + "test\\" + imgName + "+" + iteration))

    # http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    def load_model(self, path):

        # Add ops to save and restore all the variables.
        saver = tf.train.import_meta_graph('{}.ckpt.meta'.format(path))
        self.sess = tf.Session()
        # First let's load meta graph and restore weights
        path = '\\'.join(path.split("\\")[:-1]) + '\\'
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

        # get variables
        # out_chnls = 3
        # W1 = tf.get_variable("W1", shape=[5, 5, 3, out_chnls])
        # B1 = tf.get_variable("B1", shape=[out_chnls])
        # W2 = tf.get_variable("W2", shape=[5, 5, 3, out_chnls])
        # B2 = tf.get_variable("B2", shape=[out_chnls])

        graph = tf.get_default_graph()
        # self.Y2 = graph.get_tensor_by_name("Y2/Maximum:0")
        self.Y = graph.get_tensor_by_name("Y:0")
        self.input = graph.get_tensor_by_name("input:0")
        self.img_label = graph.get_tensor_by_name("img_label:0")

    def save_model(self, path):
        self.saver.save(self.sess, path + ".ckpt")


def main():
    nn = NN_Sharpen()
    tf.reset_default_graph()
    directory = '.\\data\\train\\'
    labels = []
    min = 101
    max = 200
    interval = range(min, max)
    for i in interval:
        path = os.path.join(directory, '{}.jpg'.format(i))
        labels.append(path)
    imgs = []
    for i in interval:
        path = os.path.join(directory, '{}_blur.jpg'.format(i))
        imgs.append(path)
    nn.train_on_images(imgs, labels, max - min)
    nn.save_model(".\\saved_model\\model")
    # nn.load_model(".\\saved_model\\model")
    nn.sharpen("data\\", "test4", ".jpg")


if __name__ == "__main__":
    main()
