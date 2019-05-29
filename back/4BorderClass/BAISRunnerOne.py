from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from BAISData import Data, CategoryNames
from BAISPSPNet import PSPNet
from BAISTools import Tools
from PIL import Image


class Runner(object):

    def __init__(self, log_dir, save_dir):
        self.save_dir = Tools.new_dir(save_dir)
        self.log_dir = Tools.new_dir(log_dir)

        self.last_pool_size = 90
        self.input_size = [self.last_pool_size * 8, self.last_pool_size * 8]
        pass

    def run(self, result_filename, image_filename, where=None, annotation_filename=None, ann_index=0):
        # 读入图片数据
        if annotation_filename:
            final_batch_data, data_raw, gaussian_mask, ann_data, ann_mask = Data.load_image(
                image_filename, where=where, annotation_filename=annotation_filename,
                ann_index=ann_index, image_size=self.input_size)
        else:
            final_batch_data, data_raw, gaussian_mask = Data.load_image(
                image_filename, where=where, annotation_filename=annotation_filename,
                ann_index=ann_index, image_size=self.input_size)

        # 网络
        img_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size[0], self.input_size[1], 4))
        net = PSPNet({'data': img_placeholder}, is_training=True, num_classes=21, last_pool_size=self.last_pool_size,
                     filter_number=32, num_segment=4)

        # 输出/预测
        raw_output_op = net.layers["conv6_n_4"]
        sigmoid_output_op = tf.sigmoid(raw_output_op)
        predict_output_op = tf.argmax(sigmoid_output_op, axis=-1)

        raw_output_classes = net.layers['class_attention_fc']
        pred_classes = tf.cast(tf.argmax(raw_output_classes, axis=-1), tf.int32)

        # 启动Session/加载模型
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        Tools.restore_if_y(sess, self.log_dir)

        # 运行
        raw_output, sigmoid_output, predict_output_r, raw_output_classes_r, pred_classes_r = sess.run(
            [raw_output_op, sigmoid_output_op, predict_output_op, raw_output_classes, pred_classes],
            feed_dict={img_placeholder: final_batch_data})

        # 保存
        print("{} {} {}".format(pred_classes_r[0], CategoryNames[pred_classes_r[0]], raw_output_classes_r))
        print("result in {}".format(os.path.join(self.save_dir, result_filename)))
        Image.fromarray(np.asarray(np.squeeze(data_raw), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "data.png"))

        output_result = np.squeeze(np.split(np.asarray(sigmoid_output[0] * 255, dtype=np.uint8),
                                            axis=-1, indices_or_sections=4))

        Image.fromarray(np.squeeze(np.asarray(predict_output_r[0] * 255 // 4, dtype=np.uint8))).save(
            os.path.join(self.save_dir, result_filename + "pred.png"))
        Image.fromarray(output_result[0]).save(
            os.path.join(self.save_dir, result_filename + "pred_0.png"))
        Image.fromarray(output_result[1]).save(
            os.path.join(self.save_dir, result_filename + "pred_1.png"))
        Image.fromarray(output_result[2]).save(
            os.path.join(self.save_dir, result_filename + "pred_2.png"))
        Image.fromarray(output_result[3]).save(
            os.path.join(self.save_dir, result_filename + "pred_3.png"))

        Image.fromarray(np.asarray(np.squeeze(gaussian_mask * 255), dtype=np.uint8)).save(
            os.path.join(self.save_dir, result_filename + "mask.bmp"))

        if annotation_filename:
            Image.fromarray(np.asarray(np.squeeze(ann_mask * 255), dtype=np.uint8)).save(
                os.path.join(self.save_dir, result_filename + "ann.bmp"))
            pass
        pass

    pass

if __name__ == '__main__':

    only_image = True

    if only_image:
        where = [360, 480]
        image_filename = "./input/8.jpg"
        image_name = os.path.basename(image_filename).split(".")[0]

        Runner(log_dir="./model/begin/third", save_dir="./output/begin/third/{}".format(image_name)).run(
            result_filename="{}_{}_{}_".format(image_name, where[0], where[1]),
            image_filename=image_filename, where=where)
    else:
        image_index = "2007_000063"
        where_index = 0

        Runner(log_dir="./model/begin/third", save_dir="./output/begin/third/{}".format(image_index)).run(
            result_filename="{}_{}_".format(image_index, where_index),
            image_filename="/home/z840/ALISURE/Data/VOC2012/JPEGImages/{}.jpg".format(image_index),
            annotation_filename="/home/z840/ALISURE/Data/VOC2012/SegmentationObject/{}.png".format(image_index),
            ann_index=where_index)
        pass
