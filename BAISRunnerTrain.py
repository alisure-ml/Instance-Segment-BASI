from __future__ import print_function

import os
import time

import tensorflow as tf
import tensorflow.contrib.metrics as tcm

from BAISPSPNet import PSPNet
from BAISTools import Tools
from BAISData import Data, COCOData


class Train(object):

    def __init__(self, log_dir, data, model_name="model.ckpt", is_test=False):

        # 读取数据
        self.data_reader = data
        self.batch_size = self.data_reader.batch_size
        self.num_classes = self.data_reader.num_classes
        self.input_size = self.data_reader.image_size
        self.ratio = self.data_reader.ratio
        self.num_segment = self.data_reader.num_segment
        self.attention_class = self.data_reader.attention_class

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和模型相关的参数：必须保证input_size大于8倍的last_pool_size
        self.last_pool_size = self.input_size[0] // self.ratio
        self.filter_number = 32
        self.learning_rate = 5e-3
        self.num_steps = 500001
        self.print_step = 1 if is_test else 25

        # 网络
        (self.image_placeholder, self.label_segment_placeholder, self.label_classes_placeholder,
         self.raw_output_segment, self.raw_output_classes, self.pred_segment, self.pred_classes,
         self.loss_segment, self.loss_classes, self.loss, self.accuracy_segment, self.accuracy_classes,
         self.step_ph, self.train_op, self.train_classes_op, self.learning_rate) = self.build_net()

        # summary 1
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("loss_segment", self.loss_segment)
        tf.summary.scalar("loss_classes", self.loss_classes)
        tf.summary.scalar("accuracy_segment", self.accuracy_segment)
        tf.summary.scalar("accuracy_classes", self.accuracy_classes)

        split = tf.split(self.image_placeholder, num_or_size_splits=4, axis=3)
        tf.summary.image("0-mask", split[3])
        tf.summary.image("1-image", tf.concat(split[0: 3], axis=3))
        tf.summary.image("2-label", tf.cast(self.label_segment_placeholder * (255 // (self.num_segment - 1)),
                                            dtype=tf.uint8))

        split = tf.split(self.raw_output_segment, num_or_size_splits=self.num_segment, axis=3)
        tf.summary.image("3-attention", split[self.attention_class])
        for num_segment in range(self.num_segment):
            tf.summary.image("4-segment-output-{}".format(num_segment), split[num_segment])
        tf.summary.image("5-pred_segment", tf.cast(self.pred_segment * (255 // (self.num_segment - 1)),
                                                   dtype=tf.uint8))

        self.summary_op = tf.summary.merge_all()

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        # summary 2
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        pass

    def build_net(self):
        # 数据
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size[0], self.input_size[1], 4))
        label_segment_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.input_size[0] // self.ratio,
                                                                          self.input_size[1] // self.ratio, 1))
        label_classes_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))

        # 网络
        net = PSPNet({'data': image_placeholder}, is_training=True, num_classes=self.num_classes,
                     attention_class=self.attention_class, num_segment=self.num_segment,
                     last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        raw_output_segment = net.layers['conv6_n_coco']
        raw_output_classes = net.layers['class_attention_fc']

        # Predictions
        prediction = tf.reshape(raw_output_segment, [-1, self.num_segment])
        pred_segment = tf.cast(tf.expand_dims(tf.argmax(raw_output_segment, axis=-1), axis=-1), tf.int32)
        pred_classes = tf.cast(tf.argmax(raw_output_classes, axis=-1), tf.int32)

        # label
        label_batch = tf.image.resize_nearest_neighbor(label_segment_placeholder,
                                                       tf.stack(raw_output_segment.get_shape()[1:3]))
        label_batch = tf.reshape(label_batch, [-1, ])

        # 当前批次的准确率：accuracy
        accuracy_segment = tcm.accuracy(pred_segment, label_segment_placeholder)
        accuracy_classes = tcm.accuracy(pred_classes, label_classes_placeholder)

        # loss
        loss_segment = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                                     logits=prediction))

        # 分类损失
        loss_classes = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_classes_placeholder,
                                                                                     logits=raw_output_classes))
        # 总损失
        loss = tf.add_n([loss_segment, 0.1 * loss_classes])

        # 学习率策略
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(tf.constant(self.learning_rate), tf.pow((1 - step_ph / self.num_steps), 0.9))
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # 单独训练最后的分类
        classes_trainable = [v for v in tf.trainable_variables() if 'class_attention' in v.name]
        train_classes_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=classes_trainable)

        return (image_placeholder, label_segment_placeholder, label_classes_placeholder,
                raw_output_segment, raw_output_classes, pred_segment, pred_classes,
                loss_segment, loss_classes, loss, accuracy_segment, accuracy_classes,
                step_ph, train_op, train_classes_op, learning_rate)

    def train(self, save_pred_freq, begin_step=0):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)

        for step in range(begin_step, self.num_steps):
            start_time = time.time()

            final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask = \
                self.data_reader.next_batch_train()

            # train_op = self.train_classes_op
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
                     self.loss_segment, self.loss_classes, self.loss,
                     self.raw_output_segment, self.pred_segment, self.raw_output_classes, self.pred_classes,
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
                     self.loss_segment, self.loss_classes, self.loss,
                     self.raw_output_segment, self.pred_segment, self.raw_output_classes, self.pred_classes],
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

    is_win = True
    is_voc = True

    if is_win:
        if is_voc:
            data_reader = Data(
                data_root_path="C:\\ALISURE\\DataModel\\Data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\",
                               data_list="ImageSets\\Segmentation\\train.txt",
                               data_path="JPEGImages\\", annotation_path="SegmentationObject\\",
                               class_path="SegmentationClass\\", batch_size=3, image_size=[720, 720], is_test=False)
        else:
            data_reader = COCOData(data_root_path="C:\\ALISURE\\DataModel\\Data\\COCO",
                                   data_path="", annotation_path="annotations_trainval2014\\annotations",
                                   data_type="val2014", batch_size=3, image_size=[720, 720])
            pass

        Train(log_dir="./model/begin/third", data=data_reader, is_test=True).train(save_pred_freq=2, begin_step=0)
    else:
        if is_voc:
            data_reader = Data(data_root_path="/home/z840/ALISURE/Data/VOC2012/",
                               data_list="ImageSets/Segmentation/trainval.txt",
                               data_path="JPEGImages/", annotation_path="SegmentationObject/",
                               class_path="SegmentationClass/", batch_size=3, image_size=[720, 720], is_test=False)
        else:
            data_reader = COCOData(data_root_path="/home/z840/ALISURE/Data/VOC2012/",
                                   data_path="JPEGImages/", annotation_path="SegmentationObject/",
                                   data_type="SegmentationClass/", batch_size=3, image_size=[720, 720])
            pass

        Train(log_dir="./model/begin/third", data=data_reader, is_test=False).train(save_pred_freq=2000,
                                                                                    begin_step=34001)
        pass

    pass

