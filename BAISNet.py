import numpy as np
import tensorflow as tf
from BAISTools import Tools
from nets import nets_factory


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


class LinkNet(object):

    def __init__(self, input_data, is_training, num_classes):
        self.input_data = input_data
        self.is_training = is_training
        self.num_classes = num_classes
        pass

    def _feature(self, net_input):
        self.network_fn = nets_factory.get_network_fn("vgg_16", num_classes=None,
                                                      weight_decay=0.00004, is_training=True)
        logits, end_points = self.network_fn(net_input)
        block0 = end_points['vgg_16/conv1/conv1_2']
        block1 = end_points['vgg_16/conv2/conv2_2']
        block2 = end_points['vgg_16/conv3/conv3_3']
        block3 = end_points['vgg_16/conv4/conv4_3']
        block4 = end_points['vgg_16/conv5/conv5_3']
        return block0, block1, block2, block3, block4

    def _decoder(self, net_input, input_size, output_size, name):
        with tf.variable_scope(name_or_scope=name):
            net_output = Net.conv(net_input, 1, 1, input_size[-1] // 4, 1, 1,
                                  biased=True, relu=True, padding='SAME', name='d_s_conv_1')

            net_output = tf.image.resize_nearest_neighbor(net_output, output_size[:2])
            net_output = Net.conv(net_output, 3, 3, input_size[-1] // 4, 1, 1,
                                  biased=True, relu=True, padding='SAME', name='d_s_conv_2')

            net_output = Net.conv(net_output, 1, 1, output_size[-1], 1, 1,
                                  biased=True, relu=True, name='d_s_conv_3')
            pass
        return net_output

    def _segment(self, net_input, input_size, output_size, name):
        with tf.variable_scope(name_or_scope=name):
            net_output = Net.conv(net_input, 1, 1, input_size[-1] // 4, 1, 1,
                                  biased=True, relu=True, padding='SAME', name='d_s_conv_1')

            net_output = tf.image.resize_nearest_neighbor(net_output, output_size[:2])
            net_output = Net.conv(net_output, 3, 3, input_size[-1] // 4, 1, 1,
                                  biased=True, relu=True, padding='SAME', name='d_s_conv_2')

            net_output_o = Net.conv(net_output, 1, 1, output_size[-1], 1, 1,
                                    biased=True, relu=True, name='d_s_conv_3')

            net_output = Net.conv(net_output_o, 3, 3, 2, 1, 1, biased=True, relu=False, name='d_s_conv_4')
            pass
        return net_output, net_output_o

    def build(self):

        # 提取特征，属于公共部分
        block0, block1, block2, block3, block4 = self._feature(self.input_data)
        blocks = [block0, block1, block2, block3, block4]
        block4_shape = Tools.get_shape(block4)  # 45, 512
        block3_shape = Tools.get_shape(block3)  # 90, 512
        block2_shape = Tools.get_shape(block2)  # 180, 256
        block1_shape = Tools.get_shape(block1)  # 360, 128

        adds = []
        adds_in_block = []
        adds_in_2 = []
        adds = []
        temps = []
        segments = []
        segments_output = []

        ######################################################
        # 确定初始attention的输入点：建议在进入attention时输入
        ######################################################

        with tf.variable_scope(name_or_scope="attention_4"):
            # 0
            net_segment_output, temp = self._segment(block4, block4_shape, block3_shape, name="segment_side_4")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block4, block4_shape, block3_shape, name="4")
            segments_output.append(net_output)
            pass

        with tf.variable_scope(name_or_scope="attention_3"):
            # 1  ==>
            # net_segment_block_output, temp = self._segment(block3, block3_shape, block2_shape, name="segment_side_3_block")
            # temps.append(temp)
            # segments.append(net_segment_block_output)  # segment

            # 2
            # block3_add = Net.add([block3, net_output], name='add')
            block3_add = net_output
            adds_in_block.append(block3)
            adds_in_2.append(net_output)
            adds.append(block3_add)

            net_segment_output, temp = self._segment(block3_add, block3_shape, block2_shape, name="segment_side_3")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block3_add, block3_shape, block2_shape, name="3")
            segments_output.append(net_output)
            pass

        with tf.variable_scope(name_or_scope="attention_2"):
            # 3  ==>
            # net_segment_block_output, temp = self._segment(block2, block2_shape, block1_shape, name="segment_side_2_block")
            # temps.append(temp)
            # segments.append(net_segment_block_output)  # segment

            # 4
            # block2_add = Net.add([block2, net_output], name="add")
            block2_add = net_output
            adds_in_block.append(block2)
            adds_in_2.append(net_output)
            adds.append(block2_add)

            net_segment_output, temp = self._segment(block2_add, block2_shape, block1_shape, name="segment_side_2")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block2_add, block2_shape, block1_shape, name="2")
            segments_output.append(net_output)
            pass

        with tf.variable_scope(name_or_scope="attention_1"):
            # 5  ==>
            # net_segment_block_output, temp = self._segment(block1, block1_shape, block1_shape, name="segment_side_1_block")
            # temps.append(temp)
            # segments.append(net_segment_block_output)  # segment

            # 6
            block1_add = Net.add([block1, net_output], name="add")
            adds_in_block.append(block1)
            adds_in_2.append(net_output)
            adds.append(block1_add)

            net_segment_output, temp = self._segment(block1_add, block1_shape, block1_shape, name="segment_side_1")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block1_add, block1_shape, block1_shape, name="1")
            segments_output.append(net_output)
            pass

        # 7
        net_output = Net.conv(net_output, 3, 3, 2, 1, 1, biased=True, relu=False, name='attention_0')
        segments.append(net_output)  # segment

        features = {"block": blocks, "segment": segments_output,
                    "temp": temps, "add": adds, "adds_in_block": adds_in_block, "adds_in_2": adds_in_2}
        return segments, features

    def build_old(self):

        # 提取特征，属于公共部分
        block1, block2, block3, block4 = self._feature(self.input_data)
        blocks = [block1, block2, block3, block4]
        block4_shape = Tools.get_shape(block4)  # 45, 512
        block3_shape = Tools.get_shape(block3)  # 90, 512
        block2_shape = Tools.get_shape(block2)  # 180, 256
        block1_shape = Tools.get_shape(block1)  # 360, 128

        adds = []
        adds_in_block = []
        adds_in_2 = []
        adds = []
        temps = []
        segments = []
        segments_output = []

        ######################################################
        # 确定初始attention的输入点：建议在进入attention时输入
        ######################################################

        with tf.variable_scope(name_or_scope="attention_4"):
            # 0
            net_segment_output, temp = self._segment(block4, block4_shape, block3_shape, name="segment_side_4")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block4, block4_shape, block3_shape, name="4")
            segments_output.append(net_output)
            pass

        with tf.variable_scope(name_or_scope="attention_3"):
            # 1  ==>
            # net_segment_block_output, temp = self._segment(block3, block3_shape, block2_shape, name="segment_side_3_block")
            # temps.append(temp)
            # segments.append(net_segment_block_output)  # segment

            # 2
            block3_add = Net.add([block3, net_output], name='add')
            adds_in_block.append(block3)
            adds_in_2.append(net_output)
            adds.append(block3_add)

            net_segment_output, temp = self._segment(block3_add, block3_shape, block2_shape, name="segment_side_3")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block3_add, block3_shape, block2_shape, name="3")
            segments_output.append(net_output)
            pass

        with tf.variable_scope(name_or_scope="attention_2"):
            # 3  ==>
            # net_segment_block_output, temp = self._segment(block2, block2_shape, block1_shape, name="segment_side_2_block")
            # temps.append(temp)
            # segments.append(net_segment_block_output)  # segment

            # 4
            block2_add = Net.add([block2, net_output], name="add")
            adds_in_block.append(block2)
            adds_in_2.append(net_output)
            adds.append(block2_add)

            net_segment_output, temp = self._segment(block2_add, block2_shape, block1_shape, name="segment_side_2")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block2_add, block2_shape, block1_shape, name="2")
            segments_output.append(net_output)
            pass

        with tf.variable_scope(name_or_scope="attention_1"):
            # 5  ==>
            # net_segment_block_output, temp = self._segment(block1, block1_shape, block1_shape, name="segment_side_1_block")
            # temps.append(temp)
            # segments.append(net_segment_block_output)  # segment

            # 6
            block1_add = Net.add([block1, net_output], name="add")
            adds_in_block.append(block1)
            adds_in_2.append(net_output)
            adds.append(block1_add)

            net_segment_output, temp = self._segment(block1_add, block1_shape, block1_shape, name="segment_side_1")
            temps.append(temp)
            segments.append(net_segment_output)  # segment

            net_output = self._decoder(block1_add, block1_shape, block1_shape, name="1")
            segments_output.append(net_output)
            pass

        # 7
        net_output = Net.conv(net_output, 3, 3, 2, 1, 1, biased=True, relu=False, name='attention_0')
        segments.append(net_output)  # segment

        features = {"block": blocks, "segment": segments_output,
                    "temp": temps, "add": adds, "adds_in_block": adds_in_block, "adds_in_2": adds_in_2}
        return segments, features

    pass
