import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from ai.conv.NN_Image import NN_Image


class NN_Sharpen:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 5
        self.input_width = 256
        self.input_height = 256

    def train_on_images(self, train, validate):
        start = time.time()
        self.sess = tf.Session()
        # network set-up
        self.img_label = tf.placeholder(tf.float32, shape=[1, self.input_width, self.input_height, 3], name="img_label")
        self.input = tf.placeholder(tf.float32, shape=[1, self.input_width, self.input_height, 3], name="input")
        # f_sizeX, fsizeY, in channels, out channels
        W1 = tf.get_variable("W1", initializer=tf.truncated_normal([5, 5, 3, 16], stddev=0.1))
        B1 = tf.get_variable("B1", initializer=tf.ones([16]) / 10)
        W2 = tf.get_variable("W2", initializer=tf.truncated_normal([5, 5, 3, 16], stddev=0.1))
        B2 = tf.get_variable("B2", initializer=tf.ones([3]) / 10)
        # convolution
        downsample = 2
        Y1 = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(self.input, W1,
                                                  strides=[1, downsample, downsample, 1],
                                                  padding='SAME')
                                     , B1), name="Y1")
        # deconvolution
        upsample = 2
        self.Y2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(Y1, W2,
                                                           [1, self.input_width, self.input_height, 3],
                                                           strides=[1, upsample, upsample, 1],
                                                           padding='SAME')
                                    , B2), name="Y2")
        train_cost = tf.losses.mean_squared_error(self.img_label, self.Y2)
        # train_cost = (self.img_label - Y2) ** 2
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

        i = 0
        for idx in range(len(train)):
            # prepare input and label image
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
                    output_val, loss_val, _, summaries_val = self.sess.run([self.Y2, train_cost, optim, summaries],
                                                                           feed_dict={
                                                                               self.input: img_train_reshaped,
                                                                               self.img_label: img__label_reshaped
                                                                           })
                    log.add_summary(summaries_val, i)
                    print("iter:{} loss:{}".format(i, loss_val))
                    i += 1

        print("learning took {}".format(time.time() - start))

    def show_image(self, img):
        # out_resized = img.reshape((img.shape[0], img.shape[1], 3))
        plt.imshow(img[:, :, :])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    # https://stackoverflow.com/questions/37771321/how-to-use-tensorflow-to-implement-deconvolution#37772219
    # http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    def get_bilinear_filter(self, filter_shape, upscale_factor):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                    1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init, shape=weights.shape)
        return bilinear_weights

    def upsample_layer(self, bottom, n_channels, upscale_factor):

        kernel_size = 2 * upscale_factor - upscale_factor % 2
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        with tf.variable_scope("upsample"):
            # Shape of the bottom tensor
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, n_channels]
            output_shape = tf.stack(new_shape)

            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

            weights = self.get_bilinear_filter(filter_shape, upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')

        return deconv

    def show_shapes(self, img, conv_op, deconv_op, output):
        print("""
            input (shape {})
            conv_op filters (shape {})
            deconv_op filters (shape {})
            output (shape {})
            """.format(
            img.shape, conv_op.shape, deconv_op, output))

    def sharpen(self, img):
        nn_img = NN_Image(img)
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
                # reshape image to have a leading 1 dimension
                img_shape = chunk.shape
                img_reshaped = chunk.reshape(1, img_shape[0], img_shape[1], 3)
                output_val = self.sess.run(self.Y2, feed_dict={
                    self.input: img_reshaped,
                    self.img_label: img_reshaped
                })
                # normalize for displaying
                output_val[output_val > 1.0] = 1.0
                # add chunk to final image
                output[x * chunk_x:x * chunk_x + chunk_x, y * chunk_y:y * chunk_y + chunk_y, :] = output_val
        self.show_image(output)

    # http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    def load_model(self, path):
        tf.reset_default_graph()

        # Add ops to save and restore all the variables.
        saver = tf.train.import_meta_graph('{}.ckpt.meta'.format(path))
        self.sess = tf.Session()
        # First let's load meta graph and restore weights
        path = '\\'.join(path.split("\\")[:-1]) + '\\'
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

        # get variables
        out_chnls = 3
        W1 = tf.get_variable("W1", shape=[5, 5, 3, out_chnls])
        B1 = tf.get_variable("B1", shape=[out_chnls])
        W2 = tf.get_variable("W2", shape=[5, 5, 3, out_chnls])
        B2 = tf.get_variable("B2", shape=[out_chnls])

        graph = tf.get_default_graph()
        #self.Y2 = graph.get_tensor_by_name("Y2/Maximum:0")
        self.Y2 = graph.get_tensor_by_name("Y2:0")
        self.input = graph.get_tensor_by_name("input:0")
        self.img_label = graph.get_tensor_by_name("img_label:0")

    def save_model(self, path):
        self.saver.save(self.sess, path + ".ckpt")


def main():
    nn = NN_Sharpen()
    directory = '.\\data\\train\\'
    labels = []
    for i in range(101, 150):
        path = os.path.join(directory, '{}.jpg'.format(i))
        labels.append(path)
    imgs = []
    for i in range(101, 150):
        path = os.path.join(directory, '{}_blur.jpg'.format(i))
        imgs.append(path)
    nn.train_on_images(imgs, labels)
    nn.save_model(".\\saved_model\\model")
    #nn.load_model(".\\saved_model\\model")
    nn.sharpen("data\\test.jpg")


if __name__ == "__main__":
    main()
