import csv
import math
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


class DeblurNN:

    def __init__(self):
        self.max_iter = 1000
        self.learning_rate = 0.01
        self.reg_const = 1e-9
        self.batch_size = 5
        self.img_width = 255
        self.img_heigth = 255

        self.W = {}
        self.W_cnt = 0
        self.b = []
        self.conv_info = []

        self.net_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
        self.net_label = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])

        self.input = tf.placeholder(tf.float32, shape=[None, self.img_width, self.img_heigth, 3])
        self.output = self.init_NN(self.input)

        self.stop_criterion = "file"
        self.save = True

    def init_NN(self, input):
        layer = input
        batch_size = self.batch_size
        img_width = self.img_width
        img_height = self.img_heigth

        layer = tf.reshape(layer, [batch_size, img_width, img_height, 3])

        layer = self.add_conv_layer(layer, [5, 5, 3, 16], 2, wkey="L1")
        layer = self.add_conv_layer(layer, [3, 3, 16, 32], 2, wkey="L2")
        layer = self.add_conv_layer(layer, [3, 3, 32, 32], 1, wkey="L3")
        layer = self.add_conv_layer(layer, [3, 3, 32, 32], 1, wkey="L4")
        layer = self.add_conv_layer(layer, [3, 3, 32, 32], 1, wkey="L5")

        self.encoded_image = layer

        layer = self.pop_conv_layer(layer, wkey="D1")
        layer = self.pop_conv_layer(layer, wkey="D2")
        layer = self.pop_conv_layer(layer, wkey="D3")
        layer = self.pop_conv_layer(layer, wkey="D4")
        layer = self.pop_conv_layer(layer, wkey="D5")

        layer = self.conv_layer_and_weights(layer, [3, 3, 3, 3], 1, "SAME", tf.nn.relu, wkey="K1")

        return layer

    def leaky_relu(self, x):
        return tf.maximum(0.1 * x, x)

    def add_variable(self, V, wkey=None):
        W = None
        if wkey in self.W:
            W = self.W[wkey]
        else:
            W = tf.Variable(V)
            if wkey == None:
                wkey = str(self.W_cnt)
                self.W_cnt += 1
            self.W[wkey] = W
        return W

    def add_conv_layer(self, x, filter_shape, stride, wkey=None):
        activation = self.leaky_relu
        batch_size = self.batch_size

        input_size = x.get_shape().as_list()

        W = self.add_variable(
            tf.random_normal(filter_shape, stddev=1.0 / math.sqrt(reduce(lambda x, y: x * y, filter_shape, 1.0))), wkey)
        x = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding="SAME")

        self.conv_info.append((W, input_size, stride))

        return activation(x)

    def pop_conv_layer(self, x, wkey=None):
        activation = self.leaky_relu
        batch_size = self.batch_size

        info = self.conv_info.pop()

        W = self.add_variable(tf.random_normal(info[0].get_shape().as_list(), stddev=1.0 / math.sqrt(
            reduce(lambda x, y: x * y, info[0].get_shape().as_list(), 1.0))), wkey)
        x = tf.nn.conv2d_transpose(x, W, tf.stack(info[1]), [1, info[2], info[2], 1], padding="SAME")

        return activation(x)

    def conv_layer_and_weights(self, x, conv_dims, stride, padding, activation, wkey=None):
        # mean, variance = tf.nn.moments(x, [0])
        # x = tf.nn.batch_normalization(x, mean, variance, tf.Variable(tf.random_normal(mean.get_shape().as_list())), tf.Variable(tf.random_normal(mean.get_shape().as_list())), 0.01)

        W = self.add_variable(tf.random_normal(conv_dims, stddev=1.0 / (math.sqrt(sum(conv_dims)))), wkey)
        b = None
        if wkey != None:
            b = self.add_variable(tf.random_normal([1], stddev=1.0 / (math.sqrt(sum(conv_dims)))), wkey + "_bias")
        else:
            b = self.add_variable(tf.random_normal([1], stddev=1.0 / (math.sqrt(sum(conv_dims)))))

        return activation(tf.add(tf.nn.conv2d(x, W, [1, stride, stride, 1], padding=padding), b))

    def make_file_pipeline(self, image_files, label_files, batch_size=None, im_width=100, im_height=100, shuffle=True,
                           sess=None):
        if batch_size == None:
            batch_size = self.batch_size

        # with open(image_files[0], "rb") as f:
        #    lines = f.readlines()
        #    print(lines)

        # image_files_prod = tf.train.string_input_producer(["S:\\Users\\Nibiru\\Source\\PyCharm-Projects\\Neural-Network-v2\\data\\train\\101_blur.jpg"], shuffle=shuffle, seed=1)
        # label_files_prod = tf.train.string_input_producer(["S:\\Users\\Nibiru\\Source\\PyCharm-Projects\\Neural-Network-v2\\data\\train\\101.jpg"], shuffle=shuffle, seed=1)
        image_files_prod = tf.train.string_input_producer(image_files, shuffle=shuffle, seed=1)
        label_files_prod = tf.train.string_input_producer(label_files, shuffle=shuffle, seed=1)

        reader = tf.WholeFileReader()

        image_file, image = reader.read(image_files_prod)
        label_file, label = reader.read(label_files_prod)

        decoded = tf.image.decode_jpeg(image, channels=3)
        decoded_label = tf.image.decode_jpeg(label, channels=3)

        image = decoded.eval(session=sess)  # here is your image Tensor :)
        print(image.shape)
        Image._show(Image.fromarray(np.asarray(image)))

        # image = tf.to_float() / 256.0
        # label = tf.to_float()/ 256.0

        image = tf.reshape(image, [im_width, im_height, 3])
        label = tf.reshape(label, [im_width, im_height, 3])

        image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=1000)

        return image_batch, label_batch

    def save_sanity_check(self, im_lab, sess, iteration):
        B = min(3, im_lab[0].shape[0])
        # Sanity check
        for i in range(B):
            plt.subplot(B, 3, i * 3 + 1)
            plt.imshow(im_lab[0][i])
            plt.axis("off")
            plt.subplot(B, 3, i * 3 + 2)
            plt.imshow(im_lab[1][i])
            plt.axis("off")
            plt.subplot(B, 3, i * 3 + 3)
            plt.imshow(sess.run(self.output, feed_dict={
                self.net_image: im_lab[0]
            })[i])
            plt.axis("off")
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig("iter" + str(iteration) + ".png", bbox_inches="tight")

    def train_on_images(self, train_files, label_files, val_train_files, val_label_files):
        sess = self.start_session()
        # writer = tf.summary.FileWriter("../logs/", sess.graph)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        image, label = self.make_file_pipeline(train_files, label_files, sess=sess)
        val_image, val_label = self.make_file_pipeline(val_train_files, val_label_files, sess=sess)

        cost = tf.reduce_mean((self.output - self.input) ** 2)
        # reg = tf.reduce_sum(self.W[0]*self.W[0])
        # for w in self.W[1:]:
        #    reg = reg + tf.reduce_mean(w*w)

        train_cost = cost  # + self.reg_const*reg

        # train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(train_cost)
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(train_cost)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        vl_cost = []
        tr_cost = []
        tr_iter = []

        # for iteration in range(self.num_iterations):
        iteration = 0
        while True:
            if self.stop_criterion == "iteration" and iteration >= self.max_iter:
                break
            if self.stop_criterion == "file":
                try:
                    # If stopfile exists then we stop training
                    open("stopfile")
                    break
                except IOError:
                    None

            im_lab = sess.run([image, label])

            a = sess.run(self.W["K1"])
            print(a[0, 0, 0, 0])
            sess.run(train, feed_dict={
                self.net_image: im_lab[0],
                self.net_label: im_lab[1]
            })

            if iteration % 5 == 0:
                val_im_lab = sess.run([val_image, val_label])
                vl_cost.append(sess.run(cost, feed_dict={
                    self.net_image: val_im_lab[0],
                    self.net_label: val_im_lab[1]
                }))
                tr_cost.append(sess.run(cost, feed_dict={
                    self.net_image: im_lab[0],
                    self.net_label: im_lab[1]
                }))
                tr_iter.append(iteration)
                print("Validation cost at iteration {} is {}".format(
                    iteration,
                    vl_cost[-1]
                ))
                print("Training cost at iteration {} is {}".format(
                    iteration,
                    tr_cost[-1]
                ))
            if iteration % 500 == 0:
                self.save_sanity_check(im_lab, sess, iteration)

            iteration += 1

        coord.request_stop()
        coord.join(threads)

        plt.subplot(1, 1, 1)
        plt.plot(tr_iter, tr_cost)
        plt.plot(tr_iter, vl_cost)
        plt.savefig("train_plot.png")

        with open("tr_stats.csv", "wb") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(tr_iter)
            writer.writerow(tr_cost)
            writer.writerow(vl_cost)

            if self.save == True:
                saver = tf.train.Saver()
                saver.save(sess, "../savedmodels/sharpener")

        return sess

    def start_session(self):
        sess = tf.Session()
        return sess


def test():
    filename_queue = tf.train.string_input_producer(['Climatic.png'])  # list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    my_img = tf.image.decode_png(value)  # use png or jpg decoder based on your files.

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in range(1):  # length of your filename list
        image = my_img.eval(session=sess)  # here is your image Tensor :)

    print(image.shape)
    Image._show(Image.fromarray(np.asarray(image)))

    coord.request_stop()
    coord.join(threads)


def main():
    ims = DeblurNN()

    tf.set_random_seed(5)
    # test()
    # return

    sess = ims.train_on_images(
        ["..\\..\\data\\train\\" + str(i) + "_blur.jpg" for i in range(101, 812)]
        + ["..\\..\\data\\train\\" + str(i) + "_rotated_blur.jpg" for i in range(101, 812)],
        ["..\\..\\data\\train\\" + str(i) + ".jpg" for i in range(101, 812)]
        + ["..\\..\\data\\train\\" + str(i) + "_rotated.jpg" for i in range(101, 812)],
        ["..\\..\\data\\validate\\{}_blur.jpg".format(i) for i in range(101, 200)],
        ["..\\..\\data\\validate\\{}.jpg".format(i) for i in range(101, 200)]
    )

    # sess = ims.load_model("../savedmodels/sharpener")

    # ims.sharpen(
    #    ["../data/validation_1_train" + str(i) + ".png" for i in range(10)], "_after",
    #    sess
    # )##
    # ims.test(sess)
    # ims.diagnostics(sess)


if __name__ == "__main__":
    main()
