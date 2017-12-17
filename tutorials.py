import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image


# https://www.oreilly.com/learning/hello-tensorflow
def hello_tensorflow():
    x = tf.constant(1.0, name="input")
    w = tf.Variable(0.8, name="weight")
    y = tf.multiply(x,w, name="output")
    y_ = tf.constant(0.0)
    loss = (y - y_)**2
    train_step = tf.train.AdamOptimizer(learning_rate=0.025).minimize(loss)

    for value in [x, w, y, y_, loss]:
        tf.summary.scalar(value.op.name, value)
    summaries = tf.summary.merge_all()

    sess = tf.Session()
    log = tf.summary.FileWriter('log', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        log.add_summary(sess.run(summaries), i)
        sess.run(train_step)
    print(sess.run(y))

# http://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
def playing_with_convolution():
    img = Image.open('data\\board.jpg')

    a = np.zeros([3, 3, 3, 3])
    a[1, 1, :, :] = 5
    a[0, 1, :, :] = -1
    a[1, 0, :, :] = -1
    a[2, 1, :, :] = -1
    a[1, 2, :, :] = -1

    convolve(img, a, rgb=True)

def convolve(img, kernel, strides=[1, 3, 3, 1], pooling=[1, 3, 3, 1], padding='SAME', rgb=True):
     with tf.Graph().as_default():
         num_channels = 3
         if not rgb:
             num_channels = 1
             img = img.convert('L', (0.2989, 0.5870, 0.1140, 0))  # convert to gray scale


         # reshape image to have a leading 1 dimension
         img = np.asarray(img, dtype='float32') / 256.
         img_shape = img.shape
         img_reshaped = img.reshape(1, img_shape[0], img_shape[1], num_channels)

         x = tf.placeholder('float32', [1, None, None, num_channels])
         w = tf.get_variable('w', initializer=tf.to_float(kernel))

         # operations
         conv = tf.nn.conv2d(x, w, strides=strides, padding=padding)
         sig = tf.sigmoid(conv)
         max_pool = tf.nn.max_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding=padding)
         avg_pool = tf.nn.avg_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding=padding)

         init = tf.initialize_all_variables()
         with tf.Session() as session:
             session.run(init)
             conv_op, sigmoid_op, avg_pool_op, max_pool_op = session.run([conv, sig, avg_pool, max_pool],
                                                                         feed_dict={x: img_reshaped})

         show_shapes(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op)
         if rgb:
             show_image_ops_rgb(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op)
         else:
             show_image_ops_gray(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op)

def show_image_ops_gray(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op):
     gs1 = gridspec.GridSpec(1, 5)
     plt.subplot(gs1[0, 0]); plt.axis('off'); plt.imshow(img[:, :], cmap=plt.get_cmap('gray'))
     plt.subplot(gs1[0, 1]); plt.axis('off'); plt.imshow(conv_op[0, :, :, 0], cmap=plt.get_cmap('gray'))
     plt.subplot(gs1[0, 2]); plt.axis('off'); plt.imshow(sigmoid_op[0, :, :, 0], cmap=plt.get_cmap('gray'))
     plt.subplot(gs1[0, 3]); plt.axis('off'); plt.imshow(avg_pool_op[0, :, :, 0], cmap=plt.get_cmap('gray'))
     plt.subplot(gs1[0, 4]); plt.axis('off'); plt.imshow(max_pool_op[0, :, :, 0], cmap=plt.get_cmap('gray'))
     plt.show()

def show_image_ops_rgb(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op):
    gs1 = gridspec.GridSpec(1, 5)
    plt.subplot(gs1[0, 0]); plt.axis('off'); plt.imshow(img[:, :, :])
    plt.subplot(gs1[0, 1]); plt.axis('off'); plt.imshow(conv_op[0, :, :, :])
    plt.subplot(gs1[0, 2]); plt.axis('off'); plt.imshow(sigmoid_op[0, :, :, :])
    plt.subplot(gs1[0, 3]); plt.axis('off'); plt.imshow(avg_pool_op[0, :, :, :])
    plt.subplot(gs1[0, 4]); plt.axis('off'); plt.imshow(max_pool_op[0, :, :, :])
    plt.show()

def show_shapes(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op):
     print("""
         image filters (shape {})
         conv_op filters (shape {})
         sigmoid_op filters (shape {})
         avg_pool_op filters (shape {})
         max_pool_op filters (shape {})
         """.format(
             img.shape, conv_op.shape, sigmoid_op.shape, avg_pool_op.shape, max_pool_op.shape))

def main():
    #hello_tensorflow()
    playing_with_convolution()


if __name__ == '__main__':
    main()