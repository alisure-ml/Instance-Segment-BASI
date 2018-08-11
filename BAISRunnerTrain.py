from __future__ import print_function

import os
import time

import tensorflow as tf
import tensorflow.contrib.metrics as tcm

from BAISNet import BAISNet
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
        self.num_segment = 4
        self.segment_attention = 1

        # 和模型相关的参数：必须保证input_size大于8倍的last_pool_size
        self.ratio = 8
        self.last_pool_size = last_pool_size
        self.filter_number = 32

        # 和模型训练相关的参数
        self.learning_rate = 5e-3
        self.num_steps = 500001
        self.print_step = 5 if is_test else 25

        # 读取数据
        self.data_reader = Data(data_root_path=data_root_path, data_list=train_list,
                                data_path=data_path, annotation_path=annotation_path, class_path=class_path,
                                batch_size=self.batch_size, image_size=self.input_size,
                                is_test=is_test, has_255=True)

        # 数据
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.input_size[0], self.input_size[1], 4))
        self.label_segment_placeholder = tf.placeholder(
            dtype=tf.int32, shape=(None, self.input_size[0] // self.ratio, self.input_size[1] // self.ratio, 1))
        self.label_classes_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))

        # 网络
        self.net = BAISNet(self.image_placeholder, is_training=True, num_classes=self.num_classes,
                           num_segment=self.num_segment, segment_attention=self.segment_attention,
                           last_pool_size=self.last_pool_size, filter_number=self.filter_number)

        self.segments, self.attentions, self.classes = self.net.build()
        self.final_segment_logit = self.segments[0]
        self.final_class_logit = self.classes[0]

        # Predictions
        self.pred_segment = tf.cast(tf.expand_dims(tf.argmax(self.final_segment_logit, axis=-1), axis=-1), tf.int32)
        self.pred_classes = tf.cast(tf.argmax(self.final_class_logit, axis=-1), tf.int32)

        # loss
        self.label_batch = tf.image.resize_nearest_neighbor(self.label_segment_placeholder,
                                                            tf.stack(self.final_segment_logit.get_shape()[1:3]))
        self.loss, self.loss_segment_all, self.loss_class_all, self.loss_segments, self.loss_classes = self.cal_loss(
            self.segments, self.classes, self.label_batch, self.label_classes_placeholder, self.num_segment)

        # 当前批次的准确率：accuracy
        self.accuracy_segment = tcm.accuracy(self.pred_segment, self.label_segment_placeholder)
        self.accuracy_classes = tcm.accuracy(self.pred_classes, self.label_classes_placeholder)

        with tf.name_scope("train"):
            # 学习率策略
            self.step_ph = tf.placeholder(dtype=tf.float32, shape=())
            self.learning_rate = tf.scalar_mul(tf.constant(self.learning_rate),
                                               tf.pow((1 - self.step_ph / self.num_steps), 0.9))
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # 单独训练最后的分类
            classes_trainable = [v for v in tf.trainable_variables()
                                 if 'attention_1' in v.name or "class_attention" in v.name]
            print(len(classes_trainable))
            self.train_attention_op = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(self.loss, var_list=classes_trainable)
            pass

        with tf.name_scope("summary"):
            # summary 1
            with tf.name_scope("scalar"):
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("loss_segment", self.loss_segment_all)
                tf.summary.scalar("loss_class", self.loss_class_all)
                for loss_segment_index, loss_segment in enumerate(self.loss_segments):
                    tf.summary.scalar("loss_segment_{}".format(loss_segment_index), loss_segment)
                for loss_class_index, loss_class in enumerate(self.loss_classes):
                    tf.summary.scalar("loss_class_{}".format(loss_class_index), loss_class)
                tf.summary.scalar("accuracy_segment", self.accuracy_segment)
                tf.summary.scalar("accuracy_classes", self.accuracy_classes)
                pass

            with tf.name_scope("image"):
                split = tf.split(self.image_placeholder, num_or_size_splits=4, axis=3)
                tf.summary.image("0-mask", split[3])
                tf.summary.image("1-image", tf.concat(split[0: 3], axis=3))
                tf.summary.image("2-label", tf.cast(self.label_segment_placeholder * 85, dtype=tf.uint8))
                tf.summary.image("3-pred segment", tf.cast(self.pred_segment * 85, dtype=tf.uint8))

                # attention
                for attention_index, attention in enumerate(self.attentions):
                    tf.summary.image("4-{}-attention".format(attention_index), attention)
                    pass

                for segment_index, segment in enumerate(self.segments):
                    split = tf.split(segment, num_or_size_splits=self.num_segment, axis=3)
                    tf.summary.image("5-{}-other".format(segment_index), split[0])
                    tf.summary.image("5-{}-attention".format(segment_index), split[1])
                    tf.summary.image("5-{}-border".format(segment_index), split[2])
                    tf.summary.image("5-{}-background".format(segment_index), split[-1])

                pass

            self.summary_op = tf.summary.merge_all()
            pass

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        # summary 2
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        pass

    @staticmethod
    def cal_loss(segments, classes, label_segments, label_classes, num_segment):

        label_segments = tf.reshape(label_segments, [-1, ])

        loss_segments = []
        for segment in segments:
            loss_segments.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_segments, logits=tf.reshape(segment, [-1, num_segment]))))
            pass

        loss_classes = []
        for class_one in classes:
            loss_classes.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_classes, logits=class_one)))
            pass

        loss_segment_all = tf.add_n(loss_segments) / len(loss_segments)
        loss_class_all = tf.add_n(loss_classes) / len(loss_classes)
        # 总损失
        loss = loss_segment_all + 0.1 * loss_class_all
        return loss, loss_segment_all, loss_class_all, loss_segments, loss_classes

    def train(self, save_pred_freq, begin_step=0):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)

        for step in range(begin_step, self.num_steps):
            start_time = time.time()

            final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask = \
                self.data_reader.next_batch_train()

            # train_op = self.train_attention_op
            train_op = self.train_op

            if step % self.print_step == 0:
                # summary 3
                (accuracy_segment_r, accuracy_classes_r,
                 _, learning_rate_r,
                 loss_segment_r, loss_classes_r, loss_r,
                 raw_output_r, pred_segment_r, raw_output_classes_r, pred_classes_r,
                 summary_now) = self.sess.run(
                    [self.accuracy_segment, self.accuracy_classes,
                     train_op, self.learning_rate,
                     self.loss_segment_all, self.loss_class_all, self.loss,
                     self.final_segment_logit, self.pred_segment, self.final_class_logit, self.pred_classes,
                     self.summary_op],
                    feed_dict={self.step_ph: step, self.image_placeholder: final_batch_data,
                               self.label_segment_placeholder: final_batch_ann,
                               self.label_classes_placeholder: final_batch_class})
                self.summary_writer.add_summary(summary_now, global_step=step)
            else:
                (accuracy_segment_r, accuracy_classes_r,
                 _, learning_rate_r,
                 loss_segment_r, loss_classes_r, loss_r,
                 raw_output_r, pred_segment_r, raw_output_classes_r, pred_classes_r) = self.sess.run(
                    [self.accuracy_segment, self.accuracy_classes,
                     train_op, self.learning_rate,
                     self.loss_segment_all, self.loss_class_all, self.loss,
                     self.final_segment_logit, self.pred_segment, self.final_class_logit, self.pred_classes],
                    feed_dict={self.step_ph: step, self.image_placeholder: final_batch_data,
                               self.label_segment_placeholder: final_batch_ann,
                               self.label_classes_placeholder: final_batch_class})
                pass

            if step % save_pred_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=step)
                Tools.print_info('The checkpoint has been created.')
                pass

            duration = time.time() - start_time

            Tools.print_info(
                'step {:d} loss={:.3f} seg={:.3f} class={:.3f} acc={:.3f} acc_class={:.3f}'
                ' lr={:.6f} ({:.3f} s/step) {} {}'.format(
                    step, loss_r, loss_segment_r, loss_classes_r, accuracy_segment_r, accuracy_classes_r,
                    learning_rate_r, duration, list(final_batch_class), list(pred_classes_r)))

            pass

        pass

    pass


if __name__ == '__main__':

    Train(batch_size=2, last_pool_size=90, input_size=[720, 720], log_dir="./model/attention/first",
          data_root_path="/home/z840/ALISURE/Data/VOC2012/", train_list="ImageSets/Segmentation/trainval.txt",
          data_path="JPEGImages/", annotation_path="SegmentationObject/", class_path="SegmentationClass/",
          is_test=False).train(save_pred_freq=2000, begin_step=1)

    # Train(batch_size=2, last_pool_size=30, input_size=[240, 240], log_dir="./model/begin/third",
    #       data_root_path="C:\\ALISURE\\DataModel\\Data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\",
    #       data_path="JPEGImages\\", annotation_path="SegmentationObject\\", class_path="SegmentationClass\\",
    #       train_list="ImageSets\\Segmentation\\train.txt",
    #       is_test=True).train(save_pred_freq=2, begin_step=0)
