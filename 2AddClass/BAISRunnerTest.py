from __future__ import print_function

import os
import time

import tensorflow as tf
import tensorflow.contrib.metrics as tcm

from BAISPSPNet import PSPNet
from BAISTools import Tools
from BAISData import Data


class Train(object):

    def __init__(self, batch_size, last_pool_size, input_size, log_dir,
                 data_root_path, train_list, data_path, annotation_path, class_path,
                 model_name="model.ckpt", is_test=False):

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和数据相关的参数
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = 21
        self.num_segment = 1

        # 和模型相关的参数：必须保证input_size大于8倍的last_pool_size
        self.ratio = 8
        self.last_pool_size = last_pool_size
        self.filter_number = 32

        # 读取数据
        self.data_reader = Data(data_root_path=data_root_path, data_list=train_list,
                                data_path=data_path, annotation_path=annotation_path, class_path=class_path,
                                batch_size=self.batch_size, image_size=self.input_size, is_test=is_test)
        # 网络
        self.image_placeholder, self.raw_output_segment, self.raw_output_classes, self.pred_segment, self.pred_classes = self.build_net()

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        pass

    def build_net(self):
        # 数据
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size[0], self.input_size[1], 4))

        # 网络
        net = PSPNet({'data': image_placeholder}, is_training=True, num_classes=self.num_classes,
                     num_segment=self.num_segment, last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        raw_output_segment = net.layers['conv6_n']
        raw_output_classes = net.layers['class_attention_fc']

        # Predictions
        pred_segment = tf.cast(tf.greater(raw_output_segment, 0.5), tf.int32)
        pred_classes = tf.cast(tf.argmax(raw_output_classes, axis=-1), tf.int32)

        return image_placeholder, raw_output_segment, raw_output_classes, pred_segment, pred_classes

    def train(self, save_pred_freq, begin_step=0):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)

        for step in range(begin_step, 5):

            final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask = \
                self.data_reader.next_batch_train()

            (raw_output_r, pred_segment_r, raw_output_classes_r, pred_classes_r) = self.sess.run(
                [self.raw_output_segment, self.pred_segment, self.raw_output_classes, self.pred_classes],
                feed_dict={self.image_placeholder: final_batch_data})

            if step % save_pred_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=step)
                Tools.print_info('The checkpoint has been created.')
                pass

            # Tools.print_info('step {:d} {} {} {}'.format(
            #     step, list(final_batch_class), list(pred_classes_r), list(raw_output_classes_r)))
            Tools.print_info('step {:d} {} {}'.format(step, list(final_batch_class), list(pred_classes_r)))

            pass

        pass

    pass


if __name__ == '__main__':

    Train(batch_size=8, last_pool_size=50, input_size=[400, 400], log_dir="./model/together/first",
          data_root_path="/home/z840/ALISURE/Data/VOC2012/", train_list="ImageSets/Segmentation/train.txt",
          data_path="JPEGImages/", annotation_path="SegmentationObject/", class_path="SegmentationClass/",
          is_test=True).train(save_pred_freq=2000, begin_step=1)

    # Train(batch_size=2, last_pool_size=50, input_size=[400, 400], log_dir="./model_bais/test",
    #       data_root_path="C:\\ALISURE\\DataModel\\Data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\",
    #       data_path="JPEGImages\\", annotation_path = "SegmentationObject\\", class_path = "SegmentationClass\\",
    #       train_list="ImageSets\\Segmentation\\train.txt",
    #       is_test=True).train(save_pred_freq=2, begin_step=0)

