"""
Some functions for CNN
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim  # 新增：导入 slim 库替代 contrib


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

# 修改你原本的 conv2d 函数，在定义 w 后使用 spectral_norm
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        # 核心：对权重进行谱归一化
        w = spectral_norm(w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        # 修改：使用 slim.batch_norm 替代 tf.contrib.layers.batch_norm
        # Instance Norm 不需要 decay, updates_collections 和 is_training
        # 显存太小，使用 Instance Norm 替代 Batch Norm
        return slim.instance_norm(x,
                        center=True,  # 允许学习偏移量 (beta)
                        scale=True,   # 允许学习缩放量 (gamma)，对应你之前的 scale=True
                        epsilon=self.epsilon,
                        scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    # 修改：tf.concat_v2 早已被废弃，直接使用 tf.concat 即可
    return tf.concat([
        x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


# def conv2d(input_, output_dim,
#            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#            name="conv2d"):
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#                             initializer=tf.truncated_normal_initializer(stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

#         biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#         conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

#         return conv


def gconv2d(input_, output_dim,
            k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.02,
            name="gconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for versions of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        