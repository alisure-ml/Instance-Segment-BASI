from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from BAISData import Data
from BAISPSPNet import PSPNet
from BAISTools import Tools
from PIL import Image


class Runner(object):

    def __init__(self, log_dir, save_dir):
        self.save_dir = Tools.new_dir(save_dir)
        self.log_dir = Tools.new_dir(log_dir)

        self.last_pool_size = 50
        self.input_size = [self.last_pool_size * 8, self.last_pool_size * 8]
        pass

    def run(self, result_filename, image_filename, where=None, annotation_filename=None, ann_index=0):
        # 读入图片数据
        final_batch_data, data_raw, gaussian_mask, ann_data, ann_mask = Data.load_image(
            image_filename, where=where, annotation_filename=annotation_filename,
            ann_index=ann_index, image_size=self.input_size)

        # 网络
        img_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size[0], self.input_size[1], 4))
        net = PSPNet({'data': img_placeholder}, is_training=True, num_classes=1, last_pool_size=self.last_pool_size,
                     filter_number=32)

        # 输出/预测
        raw_output_op = net.layers["conv6_n"]
        sigmoid_output_op = tf.sigmoid(raw_output_op)

        # 启动Session/加载模型
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        Tools.restore_if_y(sess, self.log_dir)

        # 运行
        raw_output, sigmoid_output = sess.run([raw_output_op, sigmoid_output_op], feed_dict={img_placeholder: final_batch_data})

        # 保存
        Image.fromarray(np.asarray(np.squeeze(data_raw), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "data.png"))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, result_filename)))
        Image.fromarray(np.asarray(np.squeeze(sigmoid_output[0] * 255), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "pred.png"))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, result_filename)))
        Image.fromarray(np.asarray(np.squeeze(np.greater(raw_output[0], 0.5) * 255), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "pred_raw.png"))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, result_filename)))
        Image.fromarray(np.asarray(np.squeeze(np.greater(sigmoid_output[0], 0.5) * 255), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "pred_sigmoid.png"))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, result_filename)))
        Image.fromarray(np.asarray(np.squeeze(gaussian_mask * 255), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "mask.bmp"))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, result_filename)))
        Image.fromarray(np.asarray(np.squeeze(ann_mask * 255), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "ann.bmp"))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, result_filename)))
        pass

    pass

if __name__ == '__main__':

    image_index = "2007_000121"
    where_index = 0
    Runner(log_dir="./model_bais/first", save_dir="./output_bais/{}".format(image_index)).run(
        result_filename="{}_{}_".format(image_index, where_index),
        image_filename="/home/z840/ALISURE/Data/VOC2012/JPEGImages/{}.jpg".format(image_index),
        annotation_filename="/home/z840/ALISURE/Data/VOC2012/SegmentationObject/{}.png".format(image_index), ann_index=where_index)
