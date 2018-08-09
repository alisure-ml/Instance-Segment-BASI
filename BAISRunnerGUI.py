import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from BAISTools import Tools
from BAISData import Data, CategoryNames
from BAISPSPNet import PSPNet


class RunnerGUI(object):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.last_pool_size = 90
        self.input_size = [self.last_pool_size * 8, self.last_pool_size * 8]

        self.sess, self.img_placeholder, self.predict_output, self.pred_classes = self.load_net()
        pass

    def load_net(self):
        print("begin to build net and start session and load model ...")

        # 网络
        img_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size[0], self.input_size[1], 4))
        net = PSPNet({'data': img_placeholder}, is_training=True, num_classes=21,
                     last_pool_size=self.last_pool_size, filter_number=32, num_segment=4)

        # 输出/预测
        raw_output_op = net.layers["conv6_n_4"]
        raw_output_op = tf.image.resize_bilinear(raw_output_op, size=self.input_size)
        predict_output_op = tf.argmax(tf.sigmoid(raw_output_op), axis=-1)

        raw_output_classes = net.layers['class_attention_fc']
        pred_classes = tf.cast(tf.argmax(raw_output_classes, axis=-1), tf.int32)

        # 启动Session/加载模型
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        Tools.restore_if_y(sess, self.log_dir)

        print("end build net and start session and load model ...")

        return sess, img_placeholder, predict_output_op, pred_classes

    def run(self, image_filename):

        plt.ion()
        plt.axis('off')

        image = np.array(Image.open(image_filename))
        plt.imshow(image)
        plt.title('Click the four extreme points of the objects')

        while 1:
            object_point = np.array(plt.ginput(1, timeout=0)).astype(np.int)[0]
            where = [int(self.input_size[0] * object_point[0] / len(image)),
                     int(self.input_size[1] * object_point[1] / len(image[0]))]
            print("point=[{},{}] where=[{},{}]".format(object_point[0], object_point[1], where[0], where[1]))

            final_batch_data, data_raw, gaussian_mask = Data.load_image(image_filename, where=where,
                                                                        image_size=self.input_size)

            print("begin to run ...")

            # 运行
            predict_output_r, pred_classes_r = self.sess.run([self.predict_output, self.pred_classes],
                                                             feed_dict={self.img_placeholder: final_batch_data})

            print("end run")

            # 类别
            print("the class is {}({})".format(pred_classes_r[0], CategoryNames[pred_classes_r[0]]))

            # 分割
            results = np.squeeze(np.asarray(predict_output_r[0] * 255 // 4, dtype=np.uint8))
            results = np.resize(results, new_shape=(len(image), len(image[0])))

            plt.imshow(results)

            # final_result = np.concatenate([image, np.expand_dims(results, axis=3)], axis=-1)
            # plt.imshow(final_result)

            pass

        pass

    pass


if __name__ == '__main__':

    RunnerGUI(log_dir="./model/begin/third").run(image_filename="./input/8.jpg")

    pass
