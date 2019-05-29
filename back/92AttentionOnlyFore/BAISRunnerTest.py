import os
import numpy as np
from PIL import Image
import tensorflow as tf
from BAISData import Data
from BAISTools import Tools
from BAISNet import LinkNet as BAISNet


class Inference(object):

    def __init__(self, input_size, log_dir, model_name="model.ckpt"):

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和数据相关的参数
        self.input_size = input_size
        self.num_classes = 21

        # 网络
        self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size[0], self.input_size[1], 3))

        # 网络
        self.net = BAISNet(self.image_placeholder, False, num_classes=self.num_classes)
        self.segments = self.net.build()
        self.pred_segment = tf.cast(tf.argmax(self.segments[0], axis=-1), dtype=tf.uint8)

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        pass

    def load_model(self):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)
        pass

    def inference(self, image_path, save_path=None):
        im_data = Data.load_data(image_path=image_path, input_size=self.input_size)
        im_data = np.expand_dims(im_data, axis=0)
        pred_segment_r = self.sess.run(self.pred_segment, feed_dict={self.image_placeholder: im_data})
        s_image = Image.fromarray(np.asarray(np.squeeze(pred_segment_r) * 255, dtype=np.uint8))
        if save_path is None:
            s_image.show()
        else:
            Tools.new_dir(save_path)
            s_image.convert("L").save("{}/{}.bmp".format(save_path, os.path.splitext(os.path.basename(image_path))[0]))
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    inference = Inference(input_size=[720, 720], log_dir="./model/segment/720")
    inference.load_model()
    data_dir = "./input"
    save_dir = "./output/segment"
    for image_name in os.listdir(data_dir):
        inference.inference(image_path="{}/{}".format(data_dir, image_name), save_path="{}".format(save_dir))
        pass
