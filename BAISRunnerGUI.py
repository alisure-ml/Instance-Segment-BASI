import os
from PIL import Image
from skimage import io as sio
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

    def run(self, image_filename_or_data, mask_color, opacity):

        plt.ion()
        plt.axis('off')

        if isinstance(image_filename_or_data, str):
            image_data = np.array(Image.open(image_filename_or_data))
        elif isinstance(image_filename_or_data, list) or isinstance(image_filename_or_data, np.ndarray):
            image_data = image_filename_or_data
        else:
            print("image_filename_or_data is error")
            return

        plt.imshow(image_data)
        plt.title('Click one point of the object that you interested')

        try:

            while 1:
                object_point = np.array(plt.ginput(1, timeout=0)).astype(np.int)[0]
                where = [int(self.input_size[0] * object_point[1] / len(image_data)),
                         int(self.input_size[1] * object_point[0] / len(image_data[0]))]
                print("point=[{},{}] where=[{},{}]".format(object_point[0], object_point[1], where[0], where[1]))

                final_batch_data, data_raw, gaussian_mask = Data.load_image(image_data, where=where,
                                                                            image_size=self.input_size)

                print("begin to run ...")

                # 运行
                predict_output_r, pred_classes_r = self.sess.run([self.predict_output, self.pred_classes],
                                                                 feed_dict={self.img_placeholder: final_batch_data})

                print("end run")

                # 类别
                print("the class is {}({})".format(pred_classes_r[0], CategoryNames[pred_classes_r[0]]))

                # 分割
                segment = np.squeeze(np.asarray(np.where(predict_output_r[0] == 1, 1, 0), dtype=np.uint8))
                segment = np.asarray(Image.fromarray(segment).resize((len(image_data[0]), len(image_data))))

                image_mask = np.ndarray(image_data.shape)
                image_mask[:, :, 0] = (1 - segment) * image_data[:, :, 0] + segment * (
                    opacity * mask_color[0] + (1 - opacity) * image_data[:, :, 0])
                image_mask[:, :, 1] = (1 - segment) * image_data[:, :, 1] + segment * (
                    opacity * mask_color[1] + (1 - opacity) * image_data[:, :, 1])
                image_mask[:, :, 2] = (1 - segment) * image_data[:, :, 2] + segment * (
                    opacity * mask_color[2] + (1 - opacity) * image_data[:, :, 2])

                plt.clf()  # clear image
                plt.text(len(image_data[0]) // 2 - 10, -6, CategoryNames[pred_classes_r[0]], fontsize=15)
                plt.imshow(image_mask.astype(np.uint8))

                print("")
                pass

        except Exception:
            print("..................")
            print("...... close .....")
            print("..................")
            pass

        pass

    pass


if __name__ == '__main__':

    # image_filename_or_data = "./input/7.jpg"
    _image_filename_or_data = np.array(Image.open("./input/7.jpg"))
    # _image_filename_or_data = sio.imread("http://farm1.staticflickr.com/169/417836491_5bf8762150_z.jpg")

    RunnerGUI(log_dir="./model/begin/third").run(_image_filename_or_data, mask_color=list([255, 0, 0]), opacity=0.5)

    pass
