import numpy as np
import tensorflow as tf


class Net(object):

    @staticmethod
    def make_var(name, shape):
        return tf.get_variable(name, shape, trainable=True)

    @staticmethod
    def zero_padding(net_input, padding, name):
        return tf.pad(net_input, paddings=np.array([[0, 0], [padding, padding], [padding, padding], [0, 0]]), name=name)

    @staticmethod
    def conv(net_input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding="VALID", biased=True):
        with tf.variable_scope(name) as scope:
            kernel = Net.make_var('weights', shape=[k_h, k_w, net_input.get_shape()[-1], c_o])
            output = tf.nn.conv2d(net_input, kernel, [1, s_h, s_w, 1], padding=padding, data_format="NHWC")
            if biased:
                biases = Net.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output
        pass

    @staticmethod
    def atrous_conv(net_input, k_h, k_w, c_o, dilation, name, relu=True, padding="VALID", biased=True):
        with tf.variable_scope(name) as scope:
            kernel = Net.make_var('weights', shape=[k_h, k_w, net_input.get_shape()[-1], c_o])
            output = tf.nn.atrous_conv2d(net_input, kernel, dilation, padding=padding)
            if biased:
                biases = Net.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output
        pass

    @staticmethod
    def relu(net_input, name):
        return tf.nn.relu(net_input, name=name)

    @staticmethod
    def sigmoid(net_input, name):
        return tf.nn.sigmoid(net_input, name=name)

    @staticmethod
    def max_pool(net_input, k_h, k_w, s_h, s_w, name, padding="VALID"):
        return tf.nn.max_pool(net_input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
                              padding=padding, name=name, data_format="NHWC")

    @staticmethod
    def avg_pool(net_input, k_h, k_w, s_h, s_w, name, padding="VALID"):
        return tf.nn.avg_pool(net_input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
                              padding=padding, name=name, data_format="NHWC")

    @staticmethod
    def lrn(net_input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(net_input, depth_radius=radius,
                                                  alpha=alpha, beta=beta, bias=bias, name=name)

    @staticmethod
    def concat(net_inputs, axis, name):
        return tf.concat(axis=axis, values=net_inputs, name=name)

    @staticmethod
    def add(net_inputs, name):
        return tf.add_n(net_inputs, name=name)

    @staticmethod
    def fc(net_input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = net_input.get_shape()
            if input_shape.ndims == 4:
                dim = input_shape[1].value * input_shape[2].value * input_shape[-1].value
                feed_in = tf.reshape(net_input, [-1, dim])
            else:
                feed_in, dim = (net_input, input_shape[-1].value)
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, Net.make_var('weights', shape=[dim, num_out]),
                    Net.make_var('biases', [num_out]), name=scope.name)
            return fc
        pass

    @staticmethod
    def softmax(net_input, name):
        input_shape = net_input.get_shape()
        if len(input_shape) > 2:
            if input_shape[1] == 1 and input_shape[2] == 1:
                return tf.squeeze(net_input, squeeze_dims=[1, 2])
            else:
                return tf.nn.softmax(net_input, name=name)
        pass

    @staticmethod
    def batch_normalization(net_input, name, relu=False, is_training=True):
        with tf.variable_scope(name) as scope:
            output = tf.layers.batch_normalization(net_input, momentum=0.95, epsilon=1e-5,
                                                   training=is_training, name=name)
            if relu:
                output = tf.nn.relu(output)
            return output
        pass

    @staticmethod
    def resize_bilinear(net_input, size, name):
        return tf.image.resize_bilinear(net_input, size=size, align_corners=True, name=name)

    @staticmethod
    def squeeze(net_inputs, name):
        return tf.squeeze(net_inputs, axis=[1, 2], name=name)

    pass


class BAISNet(object):

    def __init__(self, input_data, is_training, num_classes, num_segment,
                 segment_attention, last_pool_size, filter_number):
        self.input_data = input_data
        self.is_training = is_training
        self.num_classes = num_classes
        self.num_segment = num_segment
        self.segment_attention = segment_attention
        self.last_pool_size = last_pool_size
        self.filter_number = filter_number
        pass

    @staticmethod
    def _feature(net_input, filter_number):
        net_input = Net.conv(net_input, 3, 3, filter_number, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2_n')
        net_input = Net.batch_normalization(net_input, relu=False, name='conv1_1_3x3_s2_bn')
        net_input = Net.relu(net_input, name='conv1_1_3x3_s2_bn_relu')
        net_input = Net.conv(net_input, 3, 3, filter_number, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv1_2_3x3_bn')
        net_input = Net.conv(net_input, 3, 3, filter_number * 2, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv1_3_3x3_bn')
        net_input_pool1_3x3_s2 = Net.max_pool(net_input, 3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
        net_input = Net.conv(net_input_pool1_3x3_s2, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
        net_input_conv2_1_1x1_proj_bn = Net.batch_normalization(net_input, relu=False, name='conv2_1_1x1_proj_bn')

        net_input = Net.conv(net_input_pool1_3x3_s2, 1, 1, filter_number, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv2_1_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding1')
        net_input = Net.conv(net_input, 3, 3, filter_number, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv2_1_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
        net_input_conv2_1_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv2_1_1x1_increase_bn')

        net_input = Net.add([net_input_conv2_1_1x1_proj_bn, net_input_conv2_1_1x1_increase_bn], name='conv2_1')
        net_input_conv2_1_relu = Net.relu(net_input, name='conv2_1/relu')
        net_input = Net.conv(net_input_conv2_1_relu, 1, 1, filter_number, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv2_2_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding2')
        net_input = Net.conv(net_input, 3, 3, filter_number, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv2_2_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
        net_input_conv2_2_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv2_2_1x1_increase_bn')

        net_input = Net.add([net_input_conv2_1_relu, net_input_conv2_2_1x1_increase_bn], name='conv2_2')
        net_input_conv2_2_relu = Net.relu(net_input, name='conv2_2/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv2_3_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding3')
        net_input = Net.conv(net_input, 3, 3, filter_number, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv2_3_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
        net_input_conv2_3_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv2_3_1x1_increase_bn')

        net_input = Net.add([net_input_conv2_2_relu, net_input_conv2_3_1x1_increase_bn], name='conv2_3')
        net_input_conv2_3_relu = Net.relu(net_input, name='conv2_3/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
        net_input_conv3_1_1x1_proj_bn = Net.batch_normalization(net_input, relu=False, name='conv3_1_1x1_proj_bn')

        net_input = Net.conv(net_input_conv2_3_relu, 1, 1, filter_number * 2, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_1_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding4')
        net_input = Net.conv(net_input, 3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_1_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
        net_input_conv3_1_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv3_1_1x1_increase_bn')

        net_input = Net.add([net_input_conv3_1_1x1_proj_bn, net_input_conv3_1_1x1_increase_bn], name='conv3_1')
        net_input_conv3_1_relu = Net.relu(net_input, name='conv3_1/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_2_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding5')
        net_input = Net.conv(net_input, 3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_2_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
        net_input_conv3_2_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv3_2_1x1_increase_bn')

        net_input = Net.add([net_input_conv3_1_relu, net_input_conv3_2_1x1_increase_bn], name='conv3_2')
        net_input_conv3_2_relu = Net.relu(net_input, name='conv3_2/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_3_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding6')
        net_input = Net.conv(net_input, 3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_3_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
        net_input_conv3_3_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv3_3_1x1_increase_bn')

        net_input = Net.add([net_input_conv3_2_relu, net_input_conv3_3_1x1_increase_bn], name='conv3_3')
        net_input_conv3_3_relu = Net.relu(net_input, name='conv3_3/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_4_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=1, name='padding7')
        net_input = Net.conv(net_input, 3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv3_4_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
        net_input_conv3_4_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv3_4_1x1_increase_bn')

        net_input = Net.add([net_input_conv3_3_relu, net_input_conv3_4_1x1_increase_bn], name='conv3_4')
        net_input_conv3_4relu = Net.relu(net_input, name='conv3_4/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
        net_input_conv4_1_1x1_proj_bn = Net.batch_normalization(net_input, relu=False, name='conv4_1_1x1_proj_bn')

        net_input = Net.conv(net_input_conv3_4relu, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_1_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding8')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_1_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_1_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
        net_input_conv4_1_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_1_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_1_1x1_proj_bn, net_input_conv4_1_1x1_increase_bn], name='conv4_1')
        net_input_conv4_1_relu = Net.relu(net_input, name='conv4_1/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_2_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding9')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_2_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_2_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
        net_input_conv4_2_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_2_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_1_relu, net_input_conv4_2_1x1_increase_bn], name='conv4_2')
        net_input_conv4_2_relu = Net.relu(net_input, name='conv4_2/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_3_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding10')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_3_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_3_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
        net_input_conv4_3_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_3_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_2_relu, net_input_conv4_3_1x1_increase_bn], name='conv4_3')
        net_input_conv4_3_relu = Net.relu(net_input, name='conv4_3/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_4_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding11')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_4_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_4_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
        net_input_conv4_4_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_4_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_3_relu, net_input_conv4_4_1x1_increase_bn], name='conv4_4')
        net_input_conv4_4_relu = Net.relu(net_input, name='conv4_4/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_5_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding12')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_5_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_5_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
        net_input_conv4_5_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_5_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_4_relu, net_input_conv4_5_1x1_increase_bn], name='conv4_5')
        net_input_conv4_5_relu = Net.relu(net_input, name='conv4_5/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_6_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding13')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_6_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_6_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
        net_input_conv4_6_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_6_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_5_relu, net_input_conv4_6_1x1_increase_bn], name='conv4_6')
        net_input_conv4_6_relu = Net.relu(net_input, name='conv4_6/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_7_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_7_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding14')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_7_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_7_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_7_1x1_increase')
        net_input_conv4_7_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_7_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_6_relu, net_input_conv4_7_1x1_increase_bn], name='conv4_7')
        net_input_conv4_7_relu = Net.relu(net_input, name='conv4_7/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_8_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_8_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding15')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_8_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_8_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_8_1x1_increase')
        net_input_conv4_8_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_8_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_7_relu, net_input_conv4_8_1x1_increase_bn], name='conv4_8')
        net_input_conv4_8_relu = Net.relu(net_input, name='conv4_8/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_9_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_9_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding16')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_9_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_9_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_9_1x1_increase')
        net_input_conv4_9_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_9_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_8_relu, net_input_conv4_9_1x1_increase_bn], name='conv4_9')
        net_input_conv4_9_relu = Net.relu(net_input, name='conv4_9/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_10_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_10_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding17')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_10_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_10_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_10_1x1_increase')
        net_input_conv4_10_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_10_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_9_relu, net_input_conv4_10_1x1_increase_bn], name='conv4_10')
        net_input_conv4_10_relu = Net.relu(net_input, name='conv4_10/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_11_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_11_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding18')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_11_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_11_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_11_1x1_increase')
        net_input_conv4_11_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_11_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_10_relu, net_input_conv4_11_1x1_increase_bn], name='conv4_11')
        net_input_conv4_11_relu = Net.relu(net_input, name='conv4_11/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_12_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_12_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding19')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_12_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_12_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_12_1x1_increase')
        net_input_conv4_12_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_12_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_11_relu, net_input_conv4_12_1x1_increase_bn], name='conv4_12')
        net_input_conv4_12_relu = Net.relu(net_input, name='conv4_12/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_13_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_13_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding20')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_13_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_13_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_13_1x1_increase')
        net_input_conv4_13_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_13_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_12_relu, net_input_conv4_13_1x1_increase_bn], name='conv4_13')
        net_input_conv4_13_relu = Net.relu(net_input, name='conv4_13/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_14_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_14_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding21')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_14_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_14_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_14_1x1_increase')
        net_input_conv4_14_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_14_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_13_relu, net_input_conv4_14_1x1_increase_bn], name='conv4_14')
        net_input_conv4_14_relu = Net.relu(net_input, name='conv4_14/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_15_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_15_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding22')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_15_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_15_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_15_1x1_increase')
        net_input_conv4_15_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_15_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_14_relu, net_input_conv4_15_1x1_increase_bn], name='conv4_15')
        net_input_conv4_15_relu = Net.relu(net_input, name='conv4_15/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_16_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_16_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding23')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_16_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_16_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_16_1x1_increase')
        net_input_conv4_16_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_16_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_15_relu, net_input_conv4_16_1x1_increase_bn], name='conv4_16')
        net_input_conv4_16_relu = Net.relu(net_input, name='conv4_16/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_17_1x1_reduce')
        net_input = Net .batch_normalization(net_input, relu=True, name='conv4_17_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding24')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_17_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_17_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_17_1x1_increase')
        net_input_conv4_17_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_17_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_16_relu, net_input_conv4_17_1x1_increase_bn], name='conv4_17')
        net_input_conv4_17_relu = Net.relu(net_input, name='conv4_17/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_18_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_18_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding25')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_18_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_18_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_18_1x1_increase')
        net_input_conv4_18_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_18_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_17_relu, net_input_conv4_18_1x1_increase_bn], name='conv4_18')
        net_input_conv4_18_relu = Net.relu(net_input, name='conv4_18/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_19_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_19_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding26')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_19_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_19_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_19_1x1_increase')
        net_input_conv4_19_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_19_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_18_relu, net_input_conv4_19_1x1_increase_bn], name='conv4_19')
        net_input_conv4_19_relu = Net.relu(net_input, name='conv4_19/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_20_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_20_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding27')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_20_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_20_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_20_1x1_increase')
        net_input_conv4_20_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_20_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_19_relu, net_input_conv4_20_1x1_increase_bn], name='conv4_20')
        net_input_conv4_20_relu = Net.relu(net_input, name='conv4_20/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_21_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_21_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding28')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_21_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_21_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_21_1x1_increase')
        net_input_conv4_21_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_21_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_20_relu, net_input_conv4_21_1x1_increase_bn], name='conv4_21')
        net_input_conv4_21_relu = Net.relu(net_input, name='conv4_21/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_22_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_22_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding29')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_22_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_22_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_22_1x1_increase')
        net_input_conv4_22_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_22_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_21_relu, net_input_conv4_22_1x1_increase_bn], name='conv4_22')
        net_input_conv4_22_relu = Net.relu(net_input, name='conv4_22/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_23_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_23_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=2, name='padding30')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_23_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv4_23_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_23_1x1_increase')
        net_input_conv4_23_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv4_23_1x1_increase_bn')

        net_input = Net.add([net_input_conv4_22_relu, net_input_conv4_23_1x1_increase_bn], name='conv4_23')
        net_input_conv4_23_relu = Net.relu(net_input, name='conv4_23/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
        net_input_conv5_1_1x1_proj_bn = Net.batch_normalization(net_input, relu=False, name='conv5_1_1x1_proj_bn')

        net_input = Net.conv(net_input_conv4_23_relu, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_1_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=4, name='padding31')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 8, 4, biased=False, relu=False, name='conv5_1_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_1_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
        net_input_conv5_1_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv5_1_1x1_increase_bn')

        net_input = Net.add([net_input_conv5_1_1x1_proj_bn, net_input_conv5_1_1x1_increase_bn], name='conv5_1')
        net_input_conv5_1_relu = Net.relu(net_input, name='conv5_1/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_2_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=4, name='padding32')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 8, 4, biased=False, relu=False, name='conv5_2_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_2_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
        net_input_conv5_2_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv5_2_1x1_increase_bn')

        net_input = Net.add([net_input_conv5_1_relu, net_input_conv5_2_1x1_increase_bn], name='conv5_2')
        net_input_conv5_2_relu = Net.relu(net_input, name='conv5_2/relu')
        net_input = Net.conv(net_input, 1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_3_1x1_reduce_bn')
        net_input = Net.zero_padding(net_input, padding=4, name='padding33')
        net_input = Net.atrous_conv(net_input, 3, 3, filter_number * 8, 4, biased=False, relu=False, name='conv5_3_3x3')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_3_3x3_bn')
        net_input = Net.conv(net_input, 1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
        net_input_conv5_3_1x1_increase_bn = Net.batch_normalization(net_input, relu=False, name='conv5_3_1x1_increase_bn')

        net_input = Net.add([net_input_conv5_2_relu, net_input_conv5_3_1x1_increase_bn], name='conv5_3')
        net_input_conv5_3_relu = Net.relu(net_input, name='conv5_3/relu')
        return net_input_conv5_3_relu  # 解码器的输入

    # 输入特征，输出分割结果（多个通道）
    @staticmethod
    def _decoder(net_input_feature, filter_number, last_pool_size, num_segment):
        shape = tf.shape(net_input_feature)[1:3]
        output_filter_number = filter_number * 32 // 4

        now_size = last_pool_size // 1
        net_input = Net.avg_pool(net_input_feature, now_size, now_size, now_size, now_size, name='conv5_3_pool1')
        net_input = Net.conv(net_input, 1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_3_pool1_conv_bn')
        net_input_conv5_3_pool1_interp = Net.resize_bilinear(net_input, shape, name='conv5_3_pool1_interp')

        now_size = last_pool_size // 2
        net_input = Net.avg_pool(net_input_feature, now_size, now_size, now_size, now_size, name='conv5_3_pool2')
        net_input = Net.conv(net_input, 1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_3_pool2_conv_bn')
        net_input_conv5_3_pool2_interp = Net.resize_bilinear(net_input, shape, name='conv5_3_pool2_interp')

        now_size = last_pool_size // 3
        net_input = Net.avg_pool(net_input_feature, now_size, now_size, now_size, now_size, name='conv5_3_pool3')
        net_input = Net.conv(net_input, 1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_3_pool3_conv_bn')
        net_input_conv5_3_pool3_interp = Net.resize_bilinear(net_input, shape, name='conv5_3_pool3_interp')

        now_size = last_pool_size // 6
        net_input = Net.avg_pool(net_input_feature, now_size, now_size, now_size, now_size, name='conv5_3_pool6')
        net_input = Net.conv(net_input, 1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_3_pool6_conv_bn')
        net_input_conv5_3_pool6_interp = Net.resize_bilinear(net_input, shape, name='conv5_3_pool6_interp')

        net_input = Net.concat([net_input_feature, net_input_conv5_3_pool6_interp, net_input_conv5_3_pool3_interp,
                                net_input_conv5_3_pool2_interp, net_input_conv5_3_pool1_interp], axis=-1, name='conv5_3_concat')
        net_input = Net.conv(net_input, 3, 3, output_filter_number, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
        net_input = Net.batch_normalization(net_input, relu=True, name='conv5_4_bn')
        net_input_conv6_n_4 = Net.conv(net_input, 1, 1, num_segment, 1, 1, biased=True, relu=False, name='conv6_n_4')
        net_input_conv6_n_4_relu = Net.relu(net_input_conv6_n_4, name='conv6_n_4/relu')
        net_input_conv6_n_4_sigmoid = Net.sigmoid(net_input_conv6_n_4, name='conv6_n_4/sigmoid')
        net_input_conv6_n_4_softmax = Net.softmax(net_input_conv6_n_4, name='conv6_n_4/softmax')

        return net_input_conv6_n_4, net_input_conv6_n_4_relu, net_input_conv6_n_4_sigmoid, net_input_conv6_n_4_softmax

    # 输入特征和分割attention得分图（一个通道），输出类别
    @staticmethod
    def _classifies(net_input_multiply, last_pool_size, filter_number, num_classes):
        pool_ratio = 5
        pool_size = last_pool_size // pool_ratio
        net_input = Net.avg_pool(net_input_multiply, pool_size, pool_size, pool_size, pool_size,
                                 name="class_attention_pool")
        net_input = Net.conv(net_input, pool_ratio, pool_ratio, filter_number * 16, pool_ratio, pool_ratio,
                             name="class_attention_conv")
        net_input = Net.squeeze(net_input, name="class_attention_squeeze")
        net_input_class_attention_fc = Net.fc(net_input, num_out=num_classes, relu=False,
                                              name="class_attention_fc")
        return net_input_class_attention_fc

    @staticmethod
    def _attention():
        # with tf.variable_scope(name_or_scope="attention_2"):
        #     # attention（一个通道）
        #     attention = tf.split(segment_output, num_or_size_splits=self.num_segment, axis=3)[self.segment_attention]
        #     attentions.append(attention)
        #
        #     # 使用attention
        #     multiply = tf.multiply(net_input_feature, attention, name="class_attention_multiply")
        #
        #     # 分类
        #     class_fc = self._classifies(multiply, self.last_pool_size, self.filter_number, self.num_classes)
        #     classes.append(class_fc)
        #
        #     # 多出的部分
        #     # concat = Net.concat([net_input_feature, multiply], axis=-1, name="class_attention_concat")
        #     # net_input_feature = Net.conv(concat, 1, 1, net_input_feature.get_shape()[-1], 1, 1, name="class_attention_sample")
        #     net_input_feature = multiply * 2
        #
        #     # 解码模块，输入特征图，输出分类结果和分割结果
        #     segment_output, segment_output_relu, net_input_conv6_n_4_sigmoid = self._decoder(
        #         net_input_feature, self.filter_number, self.last_pool_size, self.num_segment)
        #     segments.append(segment_output)
        #     pass
        pass

    def build(self):
        # 提取特征，属于公共部分
        net_input_feature = self._feature(self.input_data, self.filter_number)

        segments = []
        attentions = []
        classes = []

        ######################################################
        # 确定初始attention的输入点：建议在进入attention时输入
        ######################################################

        # 解码模块，输入特征图，输出分类结果和分割结果
        segment_output, segment_output_relu, net_input_conv6_n_4_sigmoid, net_input_conv6_n_4_softmax = self._decoder(
            net_input_feature, self.filter_number, self.last_pool_size, self.num_segment)
        segments.append(segment_output)

        with tf.variable_scope(name_or_scope="attention_1"):
            # attention（一个通道）
            attention = tf.split(net_input_conv6_n_4_softmax, num_or_size_splits=self.num_segment, axis=3)[self.segment_attention]
            attentions.append(attention)

            # 使用attention
            multiply = tf.multiply(net_input_feature, attention, name="class_attention_multiply")
            net_input_feature = multiply * 2

            # 分类
            class_fc = self._classifies(multiply, self.last_pool_size, self.filter_number, self.num_classes)
            classes.append(class_fc)

            # 解码模块，输入特征图，输出分类结果和分割结果
            segment_output, segment_output_relu, net_input_conv6_n_4_sigmoid, net_input_conv6_n_4_softmax = self._decoder(
                net_input_feature, self.filter_number, self.last_pool_size, self.num_segment)
            segments.append(segment_output)
            pass

        with tf.variable_scope(name_or_scope="attention_2"):
            # attention（一个通道）
            attention = tf.split(net_input_conv6_n_4_softmax, num_or_size_splits=self.num_segment, axis=3)[self.segment_attention]
            attentions.append(attention)

            # 使用attention
            multiply = tf.multiply(net_input_feature, attention, name="class_attention_multiply")
            net_input_feature = multiply * 1

            # 分类
            class_fc = self._classifies(multiply, self.last_pool_size, self.filter_number, self.num_classes)
            classes.append(class_fc)

            # 解码模块，输入特征图，输出分类结果和分割结果
            segment_output, segment_output_relu, net_input_conv6_n_4_sigmoid, net_input_conv6_n_4_softmax = self._decoder(
                net_input_feature, self.filter_number, self.last_pool_size, self.num_segment)
            segments.append(segment_output)
            pass

        attention = tf.split(net_input_conv6_n_4_softmax, num_or_size_splits=self.num_segment, axis=3)[self.segment_attention]
        attentions.append(attention)

        # 使用attention
        multiply = tf.multiply(net_input_feature, attention, name="class_attention_multiply")

        class_fc = self._classifies(multiply, self.last_pool_size, self.filter_number, self.num_classes)
        classes.append(class_fc)

        return segments, attentions, classes

    pass
