from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from BAISData import Data, CategoryNames
from BAISNet import BAISNet
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
        net = BAISNet(img_placeholder, is_training=True, num_classes=21,
                      segment_attention=1, attention_module_num=2,
                      last_pool_size=self.last_pool_size, filter_number=32, num_segment=4)

        # 输出/预测
        op_segments, _, op_classes = net.build()

        # 启动Session/加载模型
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        Tools.restore_if_y(sess, self.log_dir)

        # Predictions
        for i in range(4):
            op_pred_classes = tf.cast(tf.argmax(op_classes[0], axis=-1), tf.int32)
            op_pred_segment = tf.cast(tf.expand_dims(tf.argmax(op_segments[0], axis=-1), axis=-1) * 85, tf.uint8)

            judge_cond = i < 2
            split = tf.split(op_segments[i], num_or_size_splits=4 if judge_cond else 2, axis=3)

            if judge_cond:
                op_other = split[0]
                op_attention = split[1]
                op_border = split[2]
                op_background = split[-1]
                r_pred_segment, r_attention, r_other, r_border, r_background, r_pred_classes = sess.run(
                    [op_pred_segment, op_attention, op_other, op_border, op_background, op_pred_classes],
                    feed_dict={img_placeholder: final_batch_data})
            else:
                op_attention = split[0]
                op_background = split[-1]
                r_pred_segment, r_attention, r_background, r_pred_classes = sess.run(
                    [op_pred_segment, op_attention, op_background, op_pred_classes],
                    feed_dict={img_placeholder: final_batch_data})
                pass

            s_image = Image.fromarray(np.asarray(np.squeeze(data_raw), dtype=np.uint8))
            s_mask = Image.fromarray(np.asarray(np.squeeze(gaussian_mask) * 255, dtype=np.uint8))
            s_pred_segment = Image.fromarray(np.asarray(np.squeeze(r_pred_segment), dtype=np.uint8))
            s_attention = Image.fromarray(np.asarray(np.squeeze(r_attention) * 255, dtype=np.uint8))
            s_background = Image.fromarray(np.asarray(np.squeeze(r_background) * 255, dtype=np.uint8))

            # 保存
            print("{} {}".format(r_pred_classes[0], CategoryNames[r_pred_classes[0]]))
            print("result in {}".format(os.path.join(self.save_dir, result_filename)))
            s_image.save(os.path.join(self.save_dir, result_filename + "{}_data.png".format(i)))
            s_pred_segment.save(os.path.join(self.save_dir, result_filename + "{}_pred_segment.png".format(i)))
            s_attention.save(os.path.join(self.save_dir, result_filename + "{}_attention.png".format(i)))
            s_background.save(os.path.join(self.save_dir, result_filename + "{}_background.png".format(i)))
            s_mask.save(os.path.join(self.save_dir, result_filename + "{}_mask.bmp".format(i)))

            if judge_cond:
                s_other = Image.fromarray(np.asarray(np.squeeze(r_other) * 255, dtype=np.uint8))
                s_border = Image.fromarray(np.asarray(np.squeeze(r_border) * 255, dtype=np.uint8))
                s_other.save(os.path.join(self.save_dir, result_filename + "{}_other.png".format(i)))
                s_border.save(os.path.join(self.save_dir, result_filename + "{}_border.png".format(i)))
            pass

        if annotation_filename:
            Image.fromarray(np.asarray(np.squeeze(ann_mask) * 255, dtype=np.uint8)).save(
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

        Runner(log_dir="./model/attention/second", save_dir="./output/attention/second/{}".format(image_name)).run(
            result_filename="{}_{}_{}_".format(image_name, where[0], where[1]),
            image_filename=image_filename, where=where)
    else:
        image_index = "2007_000063"
        where_index = 0

        Runner(log_dir="./model/attention/second", save_dir="./output/attention/second/{}".format(image_index)).run(
            result_filename="{}_{}_".format(image_index, where_index),
            image_filename="/home/z840/ALISURE/Data/VOC2012/JPEGImages/{}.jpg".format(image_index),
            annotation_filename="/home/z840/ALISURE/Data/VOC2012/SegmentationObject/{}.png".format(image_index),
            ann_index=where_index)
        pass
