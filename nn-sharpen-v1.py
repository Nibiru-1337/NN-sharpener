import os
import sys
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from ai.conv import ImageOperations

min = 0
max = 0


class NN_Sharpen:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 5
        self.input_width = 430
        self.input_height = 611
        self.layer_info = []

    def push_conv(self, name, shape, downsample, activation, in_layer):
        W = tf.get_variable("W_" + name, initializer=tf.truncated_normal(shape, stddev=0.1))
        B = tf.get_variable("B_" + name, initializer=tf.ones(shape=[shape[-1]]) / 10)
        out_layer = activation(tf.add(tf.nn.conv2d(in_layer, W,
                                                   strides=[1, downsample, downsample, 1],
                                                   padding='SAME'), B), name="CONV_" + name)
        self.layer_info.append((W, in_layer.get_shape().as_list(), downsample))
        return out_layer

    def pop_conv(self, name, shape, upsample, activation, in_layer):
        conv_shape = self.layer_info.pop()

        W = tf.get_variable("W_" + name,
                            initializer=tf.truncated_normal(conv_shape[0].get_shape().as_list(), stddev=0.1))
        B = tf.get_variable("B_" + name, initializer=tf.ones([shape[-2]]) / 10)

        layer = activation(tf.add(
            tf.nn.conv2d_transpose(in_layer, W, tf.stack(conv_shape[1]), [1, conv_shape[2], conv_shape[2], 1],
                                   padding="SAME")
            , B), name="DECONV_" + name)
        return layer

    def last_layer(self, shape, in_layer, upsample):
        W2 = tf.get_variable("W_out", initializer=tf.truncated_normal(shape, stddev=0.1))
        B2 = tf.get_variable("B_out", initializer=tf.ones([3]) / 10)
        layer = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(in_layer, W2,
                                                         [5, self.input_width, self.input_height, 3],
                                                         strides=[1, upsample, upsample, 1],
                                                         padding='SAME')
                                  , B2), name="Y")
        return layer

    def train_on_images(self, train, validate):

        image, label, name = self.make_file_pipeline(train, validate, batch_size=5)

        start = time.time()
        self.sess = tf.Session()
        # network set-up
        self.img_label = tf.placeholder(tf.float32, shape=[5, self.input_width, self.input_height, 3],
                                        name="img_label")
        self.input = tf.placeholder(tf.float32, shape=[5, self.input_width, self.input_height, 3], name="input")

        # f_sizeX, fsizeY, in channels, out channels
        layer = self.push_conv("1", [5, 5, 3, 16], 2, tf.nn.leaky_relu, self.input)
        layer = self.push_conv("2", [3, 3, 16, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.push_conv("3", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.push_conv("4", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)

        layer = self.pop_conv("d1", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.pop_conv("d2", [3, 3, 32, 32], 1, tf.nn.leaky_relu, layer)
        layer = self.pop_conv("d3", [3, 3, 16, 32], 1, tf.nn.leaky_relu, layer)
        self.Y = self.last_layer([5, 5, 3, 16], layer, 2)

        # train_cost = tf.reduce_sum(tf.square(self.Y - self.img_label))
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        indices = list(range(len(validate)))
        i = 0

        while True:
            try:
                # If stopfile exists then we stop training
                open(".\\stopfile")
                print("learning stops - took {}".format(time.time() - start))
                break
            except IOError:
                None

            im_lab = self.sess.run([image, label, name])

            self.sess.run(optim, feed_dict={
                self.input: im_lab[0],
                self.img_label: im_lab[1]
            })

            outputCost = self.sess.run(train_cost, feed_dict={
                self.input: im_lab[0],
                self.img_label: im_lab[1]
            })

            summaries_val = self.sess.run(summaries, feed_dict={
                self.input: im_lab[0],
                self.img_label: im_lab[1]
            })
            i += 1

            print(str(i))
            print(outputCost)

            log.add_summary(summaries_val, i)

            if i % 10 == 0:
                name_of_file = self.get_name(im_lab[2][0])
                self.save_image(im_lab[0], i, name_of_file)

            if i % 500 == 0:
                self.save_model(".\\saved_model\\model")

        coord.request_stop()
        coord.join(threads)

    def save_image(self, image, iter, name):

        test_image = self.sess.run(self.Y, feed_dict={
            self.input: image
        })[0]
        ImageOperations.saveFile(test_image,
                                 ".\\{}.jpg".format("data\\" + "test\\" + "blurred_iter_" + str(
                                     iter) + "_name_" + name))

    def get_name(self, name):
        label_idx = name.decode("utf-8")
        idx_of_name = label_idx.rfind('\\')
        label_idx = label_idx[idx_of_name + 1:-4]
        return label_idx

    def show_image(self, img):
        # out_resized = img.reshape((img.shape[0], img.shape[1], 3))
        plt.imshow(img[:, :, :])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def sharpen(self, path):

        filename_queue = tf.train.string_input_producer([path])

        reader = tf.WholeFileReader()

        image_file, image = reader.read(filename_queue)

        image = tf.to_float(tf.image.decode_jpeg(image, channels=3)) / 256.0

        image = tf.reshape(image, [self.input_width, self.input_height, 3])
        image_batch = tf.train.batch([image], batch_size=5, capacity=1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        test = self.sess.run(image_batch)
        self.save_image(test, -1, "sharpened")
        coord.request_stop()
        coord.join(threads)

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

    def make_file_pipeline(self, image_files, label_files, batch_size=None, shuffle=True):
        if batch_size == None:
            batch_size = self.batch_size

        image_files_prod = tf.train.string_input_producer(image_files, shuffle=shuffle, seed=1)
        label_files_prod = tf.train.string_input_producer(label_files, shuffle=shuffle, seed=1)

        reader = tf.WholeFileReader()

        image_file, image = reader.read(image_files_prod)
        label_file, label = reader.read(label_files_prod)

        image = tf.to_float(tf.image.decode_jpeg(image, channels=3)) / 256.0
        label = tf.to_float(tf.image.decode_jpeg(label, channels=3)) / 256.0

        image = tf.reshape(image, [self.input_width, self.input_height, 3])
        label = tf.reshape(label, [self.input_width, self.input_height, 3])

        image_batch, label_batch, image_file_batch = tf.train.batch([image, label, image_file], batch_size=batch_size,
                                                                    capacity=1300)

        return image_batch, label_batch, image_file_batch


def getPathsFromDir(dirPath):
    labels = []
    for x in os.walk(dirPath):
        new_list = [x for x in x[2]]
        labels = new_list

    return labels


def badSyntax():
    print("bad syntax, write --help for help")
    exit(0)


def main():
    nn = NN_Sharpen()
    test_file = ""
    load_model = ".\\saved_model\\model"
    tf.reset_default_graph()
    dir_teacher = '.\\data\\faces_resized\\'
    dir_student = '.\\data\\blurred_faces_2\\'
    num_of_images = 0
    train = False
    # i = 0
    # for path in validate_paths:
    #
    #     blurred_image = Image.open(path).filter(ImageFilter.BLUR)
    #     new_path = ".\\data\\blurred_faces\\" + str(i) + ".jpg"
    #     blurred_image.save(new_path)
    #     i = i + 1

    # train_paths = getPathsFromDir(dir_student)

    # for i in interval:
    #     new_path = ".\\data\\new_blur\\" + iter[i] + "_blurred.jpg"
    #     train_paths.append(new_path)
    validate_paths = []
    train_paths = []
    args = sys.argv
    if len(args) == 0:
        badSyntax()

    if args[1] == "--help":
        print(
            "USAGE: \n \"--method\" - can be set to train or load \n "
            "\"python nn-sharpen-v1.py --method train"
            " --dir_teacher \"absolute\\path\\to\\dir\" "
            " --dir_student \"absolute\\path\\to\\dir\"  "
            "\n where dir_teacher contains sharp images, and dir_student will contain blurry ones \n"
            "To stop learning change the name of file \"stopfile\" to something else like \"stopfile2\" \n\n"
            "\"python nn-sharpen-v1.py --method load "
            "--model \"absolute\\path\\to\\dir\" "
            "--test_file \"absolute\\path\\to\\file.jpg\" \n"
            "where --model contains already saved model of net \n"
            "and --test_file is a path to file that will be input to net")
        exit(0)

    if len(args) < 6:
        print("minimum 6 arguments needed")
        badSyntax()

    if args[1] != "--method":
        badSyntax()

    if args[2] == "train":
        if args[3] == "--dir_teacher":
            dir_teacher = args[4]
        if args[5] == "--dir_student":
            dir_student = args[6]
        else:
            badSyntax()

        train = True
        num_of_images = len(getPathsFromDir(dir_teacher))
        validate_paths = [dir_teacher + str(i) + ".jpg" for i in range(num_of_images)]
        train_paths = [dir_student + str(i) + ".jpg" for i in range(num_of_images)]
    elif args[2] == "load":
        if args[3] == "--model":
            load_model = args[4]
        if args[5] == "--test_file":
            test_file = args[6]
        else:
            badSyntax()
    else:
        badSyntax()

    if train:
        nn.train_on_images(train_paths, validate_paths)
        nn.save_model(".\\saved_model\\model")
    else:
        nn.load_model(load_model + "model")
        nn.sharpen(test_file)


if __name__ == "__main__":
    main()
