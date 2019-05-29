import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from nets import nets_factory
import tensorflow.contrib.slim as slim


CategoryNames = ['background',
                  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class Tools(object):
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_info(info):
        print("{} {}".format(time.strftime("%H:%M:%S", time.localtime()), info))
        pass

    @staticmethod
    def to_txt(data, file_name):
        with open(file_name, "w") as f:
            for one_data in data:
                f.write("{}\n".format(one_data))
            pass
        pass

    # 对输出进行着色
    @staticmethod
    def decode_labels(mask, num_images, num_classes):
        # 0 = road, 1 = sidewalk, 2 = building, 3 = wall, 4 = fence, 5 = pole,
        # 6 = traffic light, 7 = traffic sign, 8 = vegetation, 9 = terrain, 10 = sky,
        # 11 = person, 12 = rider, 13 = car, 14 = truck, 15 = bus,
        # 16 = train, 17 = motocycle, 18 = bicycle, 19 = void label
        label_colours = [(0, 0, 0), (128, 64, 128), (244, 35, 231), (69, 69, 69), (102, 102, 156), (190, 153, 153),
                         (153, 153, 153), (250, 170, 29), (219, 219, 0), (106, 142, 35), (152, 250, 152),
                         (69, 129, 180), (219, 19, 60), (255, 0, 0), (0, 0, 142), (0, 0, 69),
                         (0, 60, 100), (0, 79, 100), (0, 0, 230), (119, 10, 32), (1, 1, 1)]

        n, h, w, c = mask.shape

        assert (n >= num_images), \
            'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        return outputs

    # 对输出进行着色
    @staticmethod
    def decode_labels_test(mask, num_images, num_classes):
        # 0 = road, 1 = sidewalk, 2 = building, 3 = wall, 4 = fence, 5 = pole,
        # 6 = traffic light, 7 = traffic sign, 8 = vegetation, 9 = terrain, 10 = sky,
        # 11 = person, 12 = rider, 13 = car, 14 = truck, 15 = bus,
        # 16 = train, 17 = motocycle, 18 = bicycle, 19 = void label
        label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69), (102, 102, 156), (190, 153, 153),
                         (153, 153, 153), (250, 170, 29), (219, 219, 0), (106, 142, 35), (152, 250, 152),
                         (69, 129, 180), (219, 19, 60), (255, 0, 0), (0, 0, 142), (0, 0, 69),
                         (0, 60, 100), (0, 79, 100), (0, 0, 230), (119, 10, 32), (1, 1, 1)]
        mask = np.array(mask)
        n, h, w, c = mask.shape

        assert (n >= num_images), \
            'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        return outputs

    @staticmethod
    def prepare_label(input_batch, new_size, num_classes, one_hot=True):
        with tf.name_scope('label_encode'):
            # as labels are integer numbers, need to use NN interp.
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3])  # reducing the channel dimension.
            if one_hot:
                input_batch = tf.one_hot(input_batch, depth=num_classes)
        return input_batch

    # 如果模型存在，恢复模型
    @staticmethod
    def restore_if_y(sess, log_dir, pretrain=None):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(log_dir)
        pretrain = ckpt.model_checkpoint_path if ckpt and ckpt.model_checkpoint_path else pretrain
        if pretrain:
            # tf.train.Saver(var_list=tf.global_variables()).restore(sess, ckpt.model_checkpoint_path)
            slim.assign_from_checkpoint_fn(pretrain, var_list=tf.global_variables(), ignore_missing_vars=True)(sess)
            Tools.print_info("Restored model parameters from {}".format(pretrain))
        else:
            Tools.print_info('No checkpoint file found.')
            pass
        pass

    @staticmethod
    def get_shape(tensor):
        return [int(i) for i in list(tensor.shape)[1:]]

    pass


class Data(object):

    def __init__(self, data_list="ImageSets/Segmentation/trainval.txt", data_path="JPEGImages/",
                 data_root_path="/home/ubuntu/data1.5TB/VOC2012/",
                 annotation_path="SegmentationObject/", class_path="SegmentationClass/",
                 batch_size=4, image_size=(720, 720), is_test=False):

        self.batch_size = batch_size
        self.image_size = image_size

        # 读取数据
        self._data_list, self._annotation_list, self._class_list = self._read_list(
            data_root_path, data_list, data_path, annotation_path, class_path)

        # test
        if is_test:
            self._data_list, self._annotation_list = self._data_list[0: 200], self._annotation_list[0: 200]
            self._class_list = self._class_list[0: 200]
            pass

        # 拆解标签
        self._annotations = self._read_annotation(self._annotation_list, self.image_size)
        # 读取数据
        self._images_data = self._read_image(self._data_list, self.image_size)

        # 用来生成训练数据
        self.number_patch = len(self._annotations) // self.batch_size
        self._random_index = list(range(0, len(self._annotations)))
        self._now = 0
        pass

    def next_batch_train(self):
        # 打乱标签
        if self._now >= self.number_patch:
            print(".......................................................................")
            np.random.shuffle(self._random_index)
            self._now = 0
            pass

        # 选取当前批次的索引
        now_indexes = self._random_index[self._now * self.batch_size: (self._now + 1) * self.batch_size]

        batch_data = [self._images_data[now_index] for now_index in now_indexes]
        batch_ann = [np.expand_dims(self._annotations[now_index], axis=-1) for now_index in now_indexes]

        self._now += 1
        return batch_data, batch_ann

    @staticmethod
    def _read_annotation_old(annotation_list, image_size):
        all_ann_data = []
        for ann_index, ann_name in enumerate(annotation_list):
            # 读取数据
            ann_data = np.asarray(Image.open(ann_name).resize((image_size[0], image_size[1])))

            # 边界当背景
            ann_data = np.where(ann_data == 255, 0, ann_data)
            ann_data = np.where(ann_data > 0, 1, 0)

            # 图片id, 初始点位置，当前类别，当前掩码
            all_ann_data.append(ann_data)
            pass
        return all_ann_data

    @staticmethod
    def _read_annotation(annotation_list, image_size):
        all_ann_data = []
        for ann_index, ann_name in enumerate(annotation_list):
            # 读取数据
            ann_data = np.asarray(Image.open(ann_name).resize((image_size[0], image_size[1])))

            # 边界当背景
            ann_data = np.where(ann_data == 255, 0, ann_data)
            ann_data = np.where(ann_data > 0, 1, 0)

            # 图片id, 初始点位置，当前类别，当前掩码
            all_ann_data.append(ann_data)
            pass
        return all_ann_data

    @staticmethod
    def _read_image(data_list, image_size):

        all_data_data = []

        for data_index, data_name in enumerate(data_list):
            # 读取数据
            data_data = np.asarray(Image.open(data_name).resize(image_size), dtype=np.float32)
            # 减均值
            # data_data -= IGM_MEAN
            data_data /= 255
            all_data_data.append(data_data)
            pass

        return all_data_data

    @staticmethod
    def _read_list(data_root_path, data_list, data_path, annotation_path, class_path):
        with open(data_root_path + data_list, "r") as f:
            all_list = f.readlines()
            data_list = [data_root_path + data_path + line.strip() + ".jpg" for line in all_list]
            annotation_list = [data_root_path + annotation_path + line.strip() + ".png" for line in all_list]
            class_list = [data_root_path + class_path + line.strip() + ".png" for line in all_list]
            return data_list, annotation_list, class_list
        pass

    @staticmethod
    def load_data(image_path, input_size):
        data_data = np.asarray(Image.open(image_path).resize(input_size), dtype=np.float32) / 255
        return data_data

    pass


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
        net = slim.conv2d(net_input, 16, [3, 3], scope='conv1')
        block1 = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 16, [3, 3], scope='conv2')
        block2 = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, 32, [3, 3], scope='conv3')
        block3 = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.conv2d(net, 32, [3, 3], scope='conv4')
        block4 = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.conv2d(net, 64, [3, 3], scope='conv5')
        block5 = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool5')

        return block1, block2, block3, block4, block5

    def _decoder(self, net_input, input_size, output_size, name):
        with tf.variable_scope(name_or_scope=name):
            net_output = Net.conv(net_input, 1, 1, input_size[-1], 1, 1,
                                  biased=True, relu=True, padding='SAME', name='d_s_conv_1')
            net_output = tf.image.resize_nearest_neighbor(net_output, output_size[:2])
            net_output = Net.conv(net_output, 1, 1, output_size[-1], 1, 1,
                                  biased=True, relu=True, name='d_s_conv_3')
            pass
        return net_output

    def _segment(self, net_input, input_size, output_size, name):
        with tf.variable_scope(name_or_scope=name):
            net_output = Net.conv(net_input, 1, 1, input_size[-1], 1, 1,
                                  biased=True, relu=True, padding='SAME', name='d_s_conv_1')
            net_output = tf.image.resize_nearest_neighbor(net_output, output_size[:2])
            net_output = Net.conv(net_output, 1, 1, output_size[-1], 1, 1,
                                  biased=True, relu=True, name='d_s_conv_3')

            net_output = Net.conv(net_output, 3, 3, 2, 1, 1, biased=True, relu=False, name='d_s_conv_4')
            pass
        return net_output

    def build(self):

        # 提取特征，属于公共部分
        block1, block2, block3, block4, block5 = self._feature(self.input_data)
        blocks = [block1, block2, block3, block4, block5]
        block5_shape = Tools.get_shape(block5)  # 45, 64
        block4_shape = Tools.get_shape(block4)  # 90, 32
        block3_shape = Tools.get_shape(block3)  # 180, 16
        block2_shape = Tools.get_shape(block2)  # 360, 8
        block1_shape = Tools.get_shape(block1)  # 720, 4

        segments = []
        segments_output = []

        ######################################################
        # 确定初始attention的输入点：建议在进入attention时输入
        ######################################################

        with tf.variable_scope(name_or_scope="attention_5"):
            net_output = self._decoder(block5, block5_shape, block4_shape, name="5")
            # net_segment_output = self._segment(block5, block5_shape, block4_shape, name="segment_side_5")
            # segments.append(net_segment_output)  # segment
            pass

        segments_output.append(net_output)
        block4_add = Net.add([block4, net_output], name='attention_4_add')
        with tf.variable_scope(name_or_scope="attention_4"):
            net_output = self._decoder(block4_add, block4_shape, block3_shape, name="4")
            # net_segment_output = self._segment(block4_add, block4_shape, block3_shape, name="segment_side_4")
            # segments.append(net_segment_output)  # segment
            pass

        segments_output.append(net_output)
        block3_add = Net.add([block3, net_output], name='attention_3_add')
        with tf.variable_scope(name_or_scope="attention_3"):
            net_output = self._decoder(block3_add, block3_shape, block2_shape, name="3")
            # net_segment_output = self._segment(block3_add, block3_shape, block2_shape, name="segment_side_3")
            # segments.append(net_segment_output)  # segment
            pass

        segments_output.append(net_output)
        block2_add = Net.add([block2, net_output], name="attention_2_net_output_relu")
        with tf.variable_scope(name_or_scope="attention_2"):
            net_output = self._decoder(block2_add, block2_shape, block1_shape, name="2")
            # net_segment_output = self._segment(block2_add, block2_shape, block1_shape, name="segment_side_2")
            # segments.append(net_segment_output)  # segment
            pass

        segments_output.append(net_output)
        block1_add = Net.add([block1, net_output], name="attention_1_concat")
        with tf.variable_scope(name_or_scope="attention_1"):
            net_output = self._decoder(block1_add, block1_shape, block1_shape, name="1")
            # net_segment_output = self._segment(block1_add, block1_shape, block1_shape, name="segment_side_1")
            # segments.append(net_segment_output)  # segment
            pass

        segments_output.append(net_output)
        net_output = Net.conv(net_output, 3, 3, 2, 1, 1, biased=True, relu=False, name='attention_0')
        segments.append(net_output)  # segment

        features = {"block": blocks, "segment": segments_output}
        return segments, features

    pass


class Train(object):

    def __init__(self, batch_size, input_size, log_dir, data_root_path, train_list, data_path,
                 annotation_path, class_path, model_name="model.ckpt", pretrain=None, is_test=False):

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)
        self.pretrain = pretrain

        # 和数据相关的参数
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = 21

        # 和模型训练相关的参数
        self.learning_rate = 5e-3
        self.num_steps = 20001
        self.print_step = 10 if is_test else 10
        self.cal_step = 50 if is_test else 500

        # 读取数据
        self.data_reader = Data(data_root_path=data_root_path, data_list=train_list,
                                data_path=data_path, annotation_path=annotation_path, class_path=class_path,
                                batch_size=self.batch_size, image_size=self.input_size, is_test=is_test)

        # 数据
        self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size[0], self.input_size[1], 3))
        self.label_seg_placeholder = tf.placeholder(tf.int32, shape=(None, self.input_size[0], self.input_size[1], 1))

        # 网络
        self.net = LinkNet(self.image_placeholder, True, num_classes=self.num_classes)
        self.segments, self.features = self.net.build()

        # loss
        self.loss, self.loss_segment_all, self.loss_segments = self.cal_loss(self.segments, self.label_seg_placeholder)

        with tf.name_scope("train"):
            # 学习率策略
            self.step_ph = tf.placeholder(dtype=tf.float32, shape=())
            self.learning_rate = tf.scalar_mul(tf.constant(self.learning_rate),
                                               tf.pow((1 - self.step_ph / self.num_steps), 0.8))
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # 单独训练最后的 segment_side
            # segment_side_trainable = [v for v in tf.trainable_variables() if 'segment_side' in v.name]
            # print(len(segment_side_trainable))
            # self.train_segment_side_op = tf.train.GradientDescentOptimizer(
            #     self.learning_rate).minimize(self.loss, var_list=segment_side_trainable)
            pass

        # summary 1
        with tf.name_scope("loss"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("loss_segment", self.loss_segment_all)
            for loss_segment_index, loss_segment in enumerate(self.loss_segments):
                tf.summary.scalar("loss_segment_{}".format(loss_segment_index), loss_segment)
            pass

        with tf.name_scope("label"):
            tf.summary.image("1-image", self.image_placeholder)
            tf.summary.image("2-segment", tf.cast(self.label_seg_placeholder * 255, dtype=tf.uint8))
            pass

        with tf.name_scope("result"):
            for segment_index, segment in enumerate(self.segments):
                segment = tf.cast(tf.argmax(segment, axis=-1), dtype=tf.uint8)
                segment = tf.split(segment, num_or_size_splits=self.batch_size, axis=0)
                ii = 3 if self.batch_size >=3 else self.batch_size
                for i in range(ii):
                    tf.summary.image("predict-{}-{}".format(segment_index, i), tf.expand_dims(segment[i] * 255, axis=-1))
                    pass
                pass
            pass

        for key in list(self.features.keys()):
            with tf.name_scope(key):
                for feature_index, feature in enumerate(self.features[key]):
                    feature = tf.split(feature, num_or_size_splits=self.batch_size, axis=0)[0]
                    feature = tf.split(feature, num_or_size_splits=int(feature.shape[-1]), axis=-1)
                    for feature_one_index, feature_one in enumerate(feature):
                        tf.summary.image("{}-{}-{}".format(key, feature_index, feature_one_index), feature_one)
                    pass
                pass
            pass

        self.summary_op = tf.summary.merge_all()

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        # summary 2
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        pass

    @staticmethod
    def cal_loss(segments, label_segment):
        loss_segments = []
        for segment_index, segment in enumerate(segments):
            now_label_segment = tf.image.resize_nearest_neighbor(label_segment, tf.stack(segment.get_shape()[1:3]))

            now_loss_segment = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                targets=tf.one_hot(tf.reshape(now_label_segment, [-1, ]), depth=2),
                logits=tf.reshape(segment, [-1, 2]), pos_weight=1))

            loss_segments.append(now_loss_segment)
            pass

        loss_segment_all = tf.add_n(loss_segments) / len(loss_segments)
        # 总损失
        loss = loss_segment_all
        return loss, loss_segment_all, loss_segments

    def train(self, save_pred_freq, begin_step=0):

        # 加载模型
        # Tools.restore_if_y(self.sess, self.log_dir, pretrain=self.pretrain)

        total_loss = 0.0
        pre_avg_loss = 0.0
        for step in range(begin_step, self.num_steps):
            start_time = time.time()

            batch_data, batch_segment = self.data_reader.next_batch_train()

            train_op = self.train_op
            # train_op = self.train_segment_side_op

            if step % self.print_step == 0:
                # summary 3
                _, learning_rate_r, loss_segment_r, loss_r, summary_now = self.sess.run(
                    [train_op, self.learning_rate, self.loss_segment_all, self.loss, self.summary_op],
                    feed_dict={self.step_ph: step, self.image_placeholder: batch_data,
                               self.label_seg_placeholder: batch_segment})
                self.summary_writer.add_summary(summary_now, global_step=step)
            else:
                _, learning_rate_r, loss_segment_r, loss_r = self.sess.run(
                    [train_op, self.learning_rate, self.loss_segment_all, self.loss],
                    feed_dict={self.step_ph: step, self.image_placeholder: batch_data,
                               self.label_seg_placeholder: batch_segment})
                pass

            if step % save_pred_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=step)
                Tools.print_info('The checkpoint has been created.')
                pass

            duration = time.time() - start_time

            if step % self.cal_step == 0:
                pre_avg_loss = total_loss / self.cal_step
                total_loss = loss_r
            else:
                total_loss += loss_r

            total_loss_step = ((step % self.cal_step) + 1)

            if step % (self.cal_step // 10) == 0:
                Tools.print_info('step {:d} pre_avg_loss={:.3f} avg_loss={:.3f} loss={:.3f} seg={:.3f} lr={:.6f} '
                                 '({:.3f} s/step)'.format(step, pre_avg_loss, total_loss / total_loss_step,
                                                          loss_r, loss_segment_r, learning_rate_r, duration))
                pass
            pass

        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    Train(batch_size=16, input_size=[720, 720], log_dir="./model/segment_small/720_16",
          data_root_path="/home/ubuntu/data1.5TB/VOC2012/", train_list="ImageSets/Segmentation/trainval.txt",
          data_path="JPEGImages/", annotation_path="SegmentationObject/", class_path="SegmentationClass/",
          pretrain="/home/ubuntu/data1.5TB/ImageNetWeights/imagenet/vgg_16.ckpt",
          is_test=False).train(save_pred_freq=2000, begin_step=1)
    pass
