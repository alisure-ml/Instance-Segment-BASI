import os
import numpy as np
from PIL import Image
import tensorflow as tf
from BAISData import Data
from BAISTools import Tools
from nets import nets_factory
from BAISNet import LinkNet as BAISNet


class InferenceClass(object):

    def __init__(self, input_size, summary_dir, log_dir, model_name="model.ckpt"):

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和数据相关的参数
        self.input_size = input_size
        self.num_classes = 21

        # 网络
        self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size[0], self.input_size[1], 3))

        # 网络
        self.features = self._feature(self.image_placeholder)

        with tf.name_scope("image"):
            tf.summary.image("input", self.image_placeholder)
            pass

        with tf.name_scope("block"):
            for feature_index, feature in enumerate(self.features[:-1]):
                feature_split = tf.split(feature, num_or_size_splits=int(feature.shape[-1]), axis=-1)
                for feature_one_index, feature_one in enumerate(feature_split):
                    tf.summary.image("{}-{}".format(feature_index, feature_one_index), feature_one)
                pass
            pass

        self.summary_op = tf.summary.merge_all()

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        pass

    def load_model(self, pretrain):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir, pretrain=pretrain)
        pass

    def _feature(self, net_input):
        self.network_fn = nets_factory.get_network_fn("vgg_16", num_classes=1000,
                                                      weight_decay=0.00004, is_training=True)
        logits, end_points = self.network_fn(net_input, global_pool=True)
        block1 = end_points['vgg_16/conv2/conv2_2']
        block2 = end_points['vgg_16/conv3/conv3_3']
        block3 = end_points['vgg_16/conv4/conv4_3']
        block4 = end_points['vgg_16/conv5/conv5_3']
        result = end_points["vgg_16/fc8"]
        return [block1, block2, block3, block4, tf.nn.softmax(result, axis=-1)]

    def inference(self, image_path, image_index):
        im_data = Data.load_data(image_path=image_path, input_size=self.input_size)
        im_data = np.expand_dims(im_data, axis=0)
        result, summary_now = self.sess.run([self.features[-1], self.summary_op],
                                       feed_dict={self.image_placeholder: im_data})
        self.summary_writer.add_summary(summary_now, global_step=image_index)
        print(result)
        pass

    pass


class Inference(object):

    def __init__(self, input_size, summary_dir, log_dir, model_name="model.ckpt"):
        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和数据相关的参数
        self.input_size = input_size
        self.num_classes = 21

        # 网络
        self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size[0], self.input_size[1], 3))

        # 网络
        self.net = BAISNet(self.image_placeholder, False, num_classes=self.num_classes)
        self.segments, self.features = self.net.build()
        self.pred_segment = tf.cast(tf.argmax(self.segments[0], axis=-1), dtype=tf.uint8)

        with tf.name_scope("image"):
            tf.summary.image("input", self.image_placeholder)

            # segment
            for segment_index, segment in enumerate(self.segments):
                segment = tf.cast(tf.argmax(segment, axis=-1), dtype=tf.uint8)
                tf.summary.image("predict-{}".format(segment_index), tf.expand_dims(segment * 255, axis=-1))
                pass
            pass

        for key in list(self.features.keys()):
            with tf.name_scope(key):
                for feature_index, feature in enumerate(self.features[key]):
                    feature_split = tf.split(feature, num_or_size_splits=int(feature.shape[-1]), axis=-1)
                    for feature_one_index, feature_one in enumerate(feature_split):
                        tf.summary.image("{}-{}".format(feature_index, feature_one_index), feature_one)
                    pass
                pass

        self.summary_op = tf.summary.merge_all()

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        pass

    def load_model(self):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)
        pass

    def inference(self, image_path, image_index, save_path=None):
        im_data = Data.load_data(image_path=image_path, input_size=self.input_size)
        im_data = np.expand_dims(im_data, axis=0)
        pred_segment_r, summary_now = self.sess.run([self.pred_segment, self.summary_op],
                                                    feed_dict={self.image_placeholder: im_data})
        self.summary_writer.add_summary(summary_now, global_step=image_index)
        s_image = Image.fromarray(np.asarray(np.squeeze(pred_segment_r) * 255, dtype=np.uint8))
        if save_path is None:
            s_image.show()
        else:
            Tools.new_dir(save_path)
            s_image.convert("L").save("{}/{}.bmp".format(save_path, os.path.splitext(os.path.basename(image_path))[0]))
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_dir = "./input"
    which_image = "10"
    is_class = False
    input_size = 480
    if is_class:
        save_dir = "./output/segment_side/{}/{}_s".format(which_image, which_image)
        inference = InferenceClass(input_size=[720, 720], summary_dir=save_dir, log_dir="./model/segment_side/720")
        inference.load_model(pretrain="/home/ubuntu/data1.5TB/ImageNetWeights/imagenet/vgg_16.ckpt")
        for _image_index, image_name in enumerate(os.listdir(data_dir)):
            if which_image in image_name:
                inference.inference(image_path="{}/{}".format(data_dir, image_name), image_index=_image_index)
                pass
            pass
        pass
    else:
        save_dir = "./output/segment_add/{}/{}_ss".format(which_image, which_image)
        inference = Inference(input_size=[input_size, input_size],
                              summary_dir=save_dir, log_dir="./model/segment_add/{}".format(input_size))
        inference.load_model()
        for _image_index, image_name in enumerate(os.listdir(data_dir)):
            if which_image in image_name:
                inference.inference(image_path="{}/{}".format(data_dir, image_name),
                                    image_index=_image_index, save_path="{}".format(save_dir))
                pass
            pass
        pass

    pass
