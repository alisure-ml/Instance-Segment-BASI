import os
import time
import tensorflow as tf
from BAISData import Data
from BAISTools import Tools
from BAISNet import LinkNet as BAISNet
import tensorflow.contrib.metrics as tcm


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
        self.print_step = 10 if is_test else 100
        self.cal_step = 100 if is_test else 1000

        # 读取数据
        self.data_reader = Data(data_root_path=data_root_path, data_list=train_list,
                                data_path=data_path, annotation_path=annotation_path, class_path=class_path,
                                batch_size=self.batch_size, image_size=self.input_size, is_test=is_test)

        # 数据
        self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size[0], self.input_size[1], 3))
        self.label_seg_placeholder = tf.placeholder(tf.int32, shape=(None, self.input_size[0], self.input_size[1], 1))

        # 网络
        self.net = BAISNet(self.image_placeholder, True, num_classes=self.num_classes)
        self.segments = self.net.build()

        # loss
        self.loss, self.loss_segment_all, self.loss_segments = self.cal_loss(self.segments, self.label_seg_placeholder)

        with tf.name_scope("train"):
            # 学习率策略
            self.step_ph = tf.placeholder(dtype=tf.float32, shape=())
            self.learning_rate = tf.scalar_mul(tf.constant(self.learning_rate),
                                               tf.pow((1 - self.step_ph / self.num_steps), 0.8))
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # 单独训练最后的attention
            attention_trainable = [v for v in tf.trainable_variables()
                                   if 'attention' in v.name or "class_attention" in v.name]
            print(len(attention_trainable))
            self.train_attention_op = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(self.loss, var_list=attention_trainable)
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
            # segment
            for segment_index, segment in enumerate(self.segments):
                segment = tf.cast(tf.argmax(segment, axis=-1), dtype=tf.uint8)
                segment = tf.split(segment, num_or_size_splits=self.batch_size, axis=0)
                for i in range(self.batch_size):
                    tf.summary.image("predict-{}-{}".format(segment_index, i), tf.expand_dims(segment[i] * 255, axis=-1))
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
        Tools.restore_if_y(self.sess, self.log_dir, pretrain=self.pretrain)

        total_loss = 0.0
        pre_avg_loss = 0.0
        for step in range(begin_step, self.num_steps):
            start_time = time.time()

            batch_data, batch_segment = self.data_reader.next_batch_train()

            # train_op = self.train_attention_op
            train_op = self.train_op

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
            if step % self.print_step == 0:
                Tools.print_info("")

            pass

        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    Train(batch_size=4, input_size=[720, 720], log_dir="./model/segment/720",
          data_root_path="/home/ubuntu/data1.5TB/VOC2012/", train_list="ImageSets/Segmentation/trainval.txt",
          data_path="JPEGImages/", annotation_path="SegmentationObject/", class_path="SegmentationClass/",
          pretrain="/home/ubuntu/data1.5TB/ImageNetWeights/imagenet/vgg_16.ckpt",
          is_test=False).train(save_pred_freq=2000, begin_step=1)
