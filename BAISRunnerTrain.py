from __future__ import print_function

import os
import time

import tensorflow as tf

from BAISPSPNet import PSPNet
from BAISTools import Tools
from BAISData import Data


class Train(object):

    def __init__(self, batch_size, last_pool_size, input_size, log_dir, data_dir, train_list,
                 model_name="model.ckpt", is_test=False):

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和数据相关的参数
        self.data_dir = data_dir
        self.data_train_list = train_list
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = 21
        self.num_segment = 1

        # 和模型相关的参数：必须保证input_size大于8倍的last_pool_size
        self.ratio = 8
        self.last_pool_size = last_pool_size
        self.filter_number = 32

        # 和模型训练相关的参数
        self.learning_rate = 1e-2
        self.num_steps = 400001

        # 读取数据
        self.data_reader = Data(data_root_path=self.data_dir, data_list=self.data_train_list,
                                batch_size=self.batch_size, image_size=self.input_size, is_test=is_test)
        # 网络
        (self.image_placeholder, self.label_segment_placeholder, self.label_classes_placeholder,
         self.raw_output_segment, self.raw_output_classes, self.pred_segment, self.pred_classes,
         self.loss_segment, self.loss_classes, self.loss, self.accuracy_0, self.accuracy_1, self.accuracy_classes,
         self.step_ph, self.train_op, self.learning_rate) = self.build_net()

        # summary 1
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("loss_segment", self.loss_segment)
        tf.summary.scalar("loss_classes", self.loss_classes)
        tf.summary.scalar("accuracy_0", self.accuracy_0)
        tf.summary.scalar("accuracy_1", self.accuracy_1)
        tf.summary.scalar("accuracy_classes", self.accuracy_classes)
        split = tf.split(self.image_placeholder, num_or_size_splits=4, axis=3)
        tf.summary.image("image", tf.concat(split[0: 3], axis=3))
        tf.summary.image("mask", split[3])
        tf.summary.image("label", self.label_segment_placeholder)
        tf.summary.image("raw_output_segment", self.raw_output_segment)
        tf.summary.image("pred_segment", tf.cast(self.pred_segment, tf.float32))
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
        label_segment_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size[0] // self.ratio,
                                                                            self.input_size[1] // self.ratio, 1))
        label_classes_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))

        # 网络
        net = PSPNet({'data': image_placeholder}, is_training=True, num_classes=self.num_classes,
                     num_segment=self.num_segment, last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        raw_output_segment = net.layers['conv6_n']
        raw_output_classes = net.layers['class_1']

        # Predictions
        prediction = tf.reshape(raw_output_segment, [-1, ])
        pred_segment = tf.cast(tf.greater(raw_output_segment, 0.5), tf.int32)
        pred_classes = tf.argmax(raw_output_classes, axis=-1, output_type=tf.int32)

        # label
        label_batch = tf.image.resize_nearest_neighbor(label_segment_placeholder,
                                                       tf.stack(raw_output_segment.get_shape()[1:3]))
        label_batch = tf.cast(tf.reshape(label_batch, [-1, ]), tf.float32)

        # 当前批次的准确率：accuracy
        accuracy_1 = tf.reduce_mean(tf.cast(tf.cast(tf.greater(label_batch, 0.5), tf.int32), tf.float32))
        accuracy_0 = tf.reduce_mean(tf.cast(tf.cast(tf.greater(prediction, 0.5), tf.int32), tf.float32))
        accuracy_classes = tf.reduce_mean(tf.cast(tf.equal(pred_classes, label_classes_placeholder), tf.int32))

        # loss
        loss_segment = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label_batch,
                                                                               logits=prediction, pos_weight=3))
        # 分类损失
        loss_classes = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_classes_placeholder,
                                                                                     logits=raw_output_classes))
        # 总损失
        loss = tf.add_n([loss_segment, loss_classes])

        # Poly learning rate policy
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(tf.constant(self.learning_rate), tf.pow((1 - step_ph / self.num_steps), 0.9))
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        return (image_placeholder, label_segment_placeholder, label_classes_placeholder,
                raw_output_segment, raw_output_classes, pred_segment, pred_classes,
                loss_segment, loss_classes, loss, accuracy_0, accuracy_1, accuracy_classes,
                step_ph, train_op, learning_rate)

    def train(self, save_pred_freq, begin_step=0):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)

        for step in range(begin_step, self.num_steps):
            start_time = time.time()

            final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask = \
                self.data_reader.next_batch_train()

            if step % 50 == 0:
                # summary 3
                (accuracy_0_r, accuracy_1_r, accuracy_classes_r,
                 _, learning_rate_r,
                 loss_segment_r, loss_classes_r, loss_r,
                 raw_output_r, pred_segment_r, raw_output_classes_r, pred_classes_r,
                 summary_now) = self.sess.run(
                    [self.accuracy_0, self.accuracy_1, self.accuracy_classes,
                     self.train_op, self.learning_rate,
                     self.loss_segment, self.loss_classes, self.loss,
                     self.raw_output_segment, self.pred_segment, self.raw_output_classes, self.pred_classes,
                     self.summary_op],
                    feed_dict={self.step_ph: step, self.image_placeholder: final_batch_data,
                               self.label_segment_placeholder: final_batch_ann,
                               self.label_classes_placeholder: final_batch_class})
                self.summary_writer.add_summary(summary_now, global_step=step)
            else:
                (accuracy_0_r, accuracy_1_r, accuracy_classes_r,
                 _, learning_rate_r,
                 loss_segment_r, loss_classes_r, loss_r,
                 raw_output_r, pred_segment_r, raw_output_classes_r, pred_classes_r) = self.sess.run(
                    [self.accuracy_0, self.accuracy_1, self.accuracy_classes,
                     self.train_op, self.learning_rate,
                     self.loss_segment, self.loss_classes, self.loss,
                     self.raw_output_segment, self.pred_segment, self.raw_output_classes, self.pred_classes],
                    feed_dict={self.step_ph: step, self.image_placeholder: final_batch_data,
                               self.label_segment_placeholder: final_batch_ann,
                               self.label_classes_placeholder: final_batch_class})
                pass

            if step % save_pred_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=step)
                Tools.print_info('The checkpoint has been created.')
            duration = time.time() - start_time
            Tools.print_info(
                'step {:d} loss={:.6f} loss_segment={:.6f} loss_classes={:.6f}'
                ' acc_0={:.6f} acc_1={:.6f} acc_classes={:.6f}'
                ' learning_rate={:.6f} ({:.6f} sec/step)'.format(
                    step, loss_r, loss_segment_r, loss_classes_r,
                    accuracy_0_r, accuracy_1_r, accuracy_classes_r, learning_rate_r, duration))

            pass
        pass

    pass


if __name__ == '__main__':

    # Train(batch_size=8, last_pool_size=50, input_size=[400, 400], log_dir="./model_bais/first",
    #       data_dir="/home/z840/ALISURE/Data/VOC2012/", train_list="ImageSets/Segmentation/train.txt",
    #       is_test=False).train(save_pred_freq=2000, begin_step=0)
    Train(batch_size=2, last_pool_size=50, input_size=[400, 400], log_dir="./model_bais/test",
          data_dir="C:\\ALISURE\\DataModel\\Data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\",
          train_list="ImageSets\\Segmentation\\train.txt",
          is_test=True).train(save_pred_freq=2, begin_step=0)
