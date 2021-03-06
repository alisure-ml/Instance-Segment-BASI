import os
import numpy as np
import xml.etree.ElementTree as et
from PIL import Image, ImageDraw, ImageFont


# IGM_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

CategoryNames = ['background',
                  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class Data(object):

    def __init__(self, data_list="ImageSets\\Segmentation\\train.txt", data_path="JPEGImages\\",
                 data_root_path="C:\\ALISURE\\DataModel\\Data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\",
                 annotation_path="SegmentationObject\\", class_path="SegmentationClass\\",
                 batch_size=4, image_size=(720, 720), ratio=8, is_test=False):

        self.batch_size = batch_size
        self.image_size = image_size
        self.ratio = ratio

        # 读取数据
        self._data_list, self._annotation_list, self._class_list = self._read_list(
            data_root_path, data_list, data_path, annotation_path, class_path)

        # test
        if is_test:
            self._data_list, self._annotation_list = self._data_list[0: 12], self._annotation_list[0: 12]
            self._class_list = self._class_list[0: 12]
            pass

        # 拆解标签
        self._annotations = self._read_annotation(self._annotation_list, self._class_list, self.image_size, self.ratio)
        # 读取数据
        self._images_data = self._read_image(self._data_list, self.image_size)

        # 用来生成训练数据
        self.number_patch = len(self._annotations) // self.batch_size
        self._random_index = list(range(0, len(self._annotations)))
        self._now = 0
        pass

    def next_batch_train(self):
        # 打乱标签
        if self._now >= self.number_patch:
            print(".......................................................................")
            np.random.shuffle(self._random_index)
            self._now = 0
            pass

        # 选取当前批次的索引
        now_indexes = self._random_index[self._now * self.batch_size: (self._now + 1) * self.batch_size]

        # 标注
        batch_ann = [self._annotations[now_index] for now_index in now_indexes]

        # 选取初始点的位置
        for ann_index, ann in enumerate(batch_ann):
            where = np.argwhere(ann[-1] == 1)
            where = where[np.random.randint(0, len(where))]
            batch_ann[ann_index][1] = [where[0] * self.ratio, where[1] * self.ratio]
            pass

        # 根据初始点生成高斯Mask
        batch_mask = []
        for ann_index, ann in enumerate(batch_ann):
            batch_mask.append(self._mask_gaussian(self.image_size, ann[1]))
            pass

        # 数据
        batch_data = [self._images_data[ann[0]] for ann in batch_ann]

        # 数据+MASK
        final_batch_data = [np.concatenate((one_data, np.expand_dims(one_mask, 2)), 2)
                            for one_data, one_mask in zip(batch_data, batch_mask)]

        # 标注
        final_batch_ann = [np.expand_dims(one_ann[-1], 2) for one_ann in batch_ann]

        # 类别
        final_batch_class = [one_ann[2] for one_ann in batch_ann]

        # save annotation
        # for one_index, one_ann in enumerate(batch_ann):
        #     Image.fromarray(np.asarray(one_ann[-1] * 255, dtype=np.uint8)).convert("L").save("{}.bmp".format(one_index))
        #     pass

        self._now += 1
        return final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask

    def next_batch_test(self, batch_index):
        if batch_index >= self.number_patch:
            print("there is error, because that test is over .....")

        # 标注
        batch_ann = self._annotations[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        # 选取初始点的位置
        for ann_index, ann in enumerate(batch_ann):
            where = np.argwhere(ann[-1] == 1)
            batch_ann[ann_index][1] = where[np.random.randint(0, len(where))]
            pass

        # 根据初始点生成高斯Mask
        batch_mask = []
        for ann_index, ann in enumerate(batch_ann):
            batch_mask.append(self._mask_gaussian(self.image_size, ann[1]))
            pass

        # 数据
        batch_data = [self._images_data[ann[0]] for ann in batch_ann]

        # 数据+MASK
        final_batch_data = [np.concatenate((one_data, np.expand_dims(one_mask, 2)), 2)
                            for one_data, one_mask in zip(batch_data, batch_mask)]

        # 标注
        final_batch_ann = [np.expand_dims(one_ann[-1], 2) for one_ann in batch_ann]

        # 类别
        final_batch_class = [one_ann[2] for one_ann in batch_ann]

        return final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask

    @staticmethod
    def _read_annotation(annotation_list, class_list, image_size, ratio):

        all_ann_data = []

        for ann_index, ann_name in enumerate(annotation_list):
            # 读取数据
            class_data = np.asarray(Image.open(class_list[ann_index]).resize(
                (image_size[0]//ratio, image_size[1]//ratio)))
            ann_data = np.asarray(Image.open(ann_name).resize((image_size[0]//ratio, image_size[1]//ratio)))

            # 所有的标注数字：其中0为背景，255为边界，其他为物体
            nums = [i for i in range(1, 255) if np.any(ann_data == i)]

            # 所有标签的掩码和类别
            ann_mask = []
            ann_class = []
            for num in nums:
                ann_mask.append(np.where(ann_data == num, 1, 0))
                # 类别信息
                where_num = np.where(ann_data == num)
                class_num = class_data[where_num[0][0]][where_num[1][0]]
                class_num = 0 if class_num >= len(CategoryNames) else class_num
                ann_class.append(class_num)
                pass

            # 图片id, 初始点位置，当前类别，当前掩码
            [all_ann_data.append([ann_index, num, ann_class[num_index],
                                  ann_mask[num_index]]) for num_index, num in enumerate(nums)]
            pass

        return all_ann_data

    @staticmethod
    def _read_image(data_list, image_size):

        all_data_data = []

        for data_index, data_name in enumerate(data_list):
            # 读取数据
            data_data = np.asarray(Image.open(data_name).resize(image_size), dtype=np.float32)
            # 减均值
            # data_data -= IGM_MEAN
            data_data /= 255
            all_data_data.append(data_data)
            pass

        return all_data_data

    @staticmethod
    def _read_list(data_root_path, data_list, data_path, annotation_path, class_path):
        with open(data_root_path + data_list, "r") as f:
            all_list = f.readlines()
            data_list = [data_root_path + data_path + line.strip() + ".jpg" for line in all_list]
            annotation_list = [data_root_path + annotation_path + line.strip() + ".png" for line in all_list]
            class_list = [data_root_path + class_path + line.strip() + ".png" for line in all_list]
            return data_list, annotation_list, class_list
        pass

    @staticmethod
    def _mask_gaussian(image_size, where, sigma=30):

        x = np.arange(0, image_size[1], 1, float)
        y = np.arange(0, image_size[0], 1, float)
        y = y[:, np.newaxis]

        x0 = where[1]
        y0 = where[0]

        # 生成高斯掩码
        mask = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(np.float32)

        return mask

    # 测试时使用
    @staticmethod
    def load_image(image_filename, where=None, annotation_filename=None, ann_index=0, image_size=(720, 720)):

        # 读取数据
        data_raw = np.asarray(Image.open(image_filename).resize(image_size), dtype=np.float32)
        # 减均值
        # data_data = data_raw - IGM_MEAN
        data_data = data_raw / 255

        if annotation_filename is not None:
            # 读取数据
            ann_data = np.asarray(Image.open(annotation_filename).resize(image_size))
            # 所有的标注数字：其中0为背景，255为边界，其他为物体
            nums = [i for i in range(1, 255) if np.any(ann_data == i)]
            # 所有标签的掩码
            ann_mask = [np.where(ann_data == i, 1, 0) for i in nums]
            pass

            # 选取初始点的位置
            where = np.argwhere(ann_mask[ann_index] == 1)
            where = where[np.random.randint(0, len(where))]

            if where is None:
                raise Exception("where can not none")
                pass

            # 根据初始点生成高斯Mask
            gaussian_mask = Data._mask_gaussian(image_size, where)

            # 数据+MASK
            final_batch_data = [np.concatenate((data_data, np.expand_dims(gaussian_mask, 2)), 2)]

            return final_batch_data, data_raw, gaussian_mask, ann_data, ann_mask[ann_index]
        else:
            if where is None:
                raise Exception("where can not none")
                pass

            # 根据初始点生成高斯Mask
            gaussian_mask = Data._mask_gaussian(image_size, where)

            # 数据+MASK
            final_batch_data = [np.concatenate((data_data, np.expand_dims(gaussian_mask, 2)), 2)]

            return final_batch_data, data_raw, gaussian_mask
        pass

    pass


class Annotation(object):

    def __init__(self,
                 data_list="ImageSets\\Segmentation\\train.txt", data_path="JPEGImages\\",
                 data_root_path="C:\\ALISURE\\DataModel\\Data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\",
                 annotation_path="SegmentationObject\\", xml_path="Annotations\\"):

        data_list, annotation_list = Data._read_list(data_root_path, data_list, data_path, annotation_path)

        for image_file_name in annotation_list:
            xml_file_name = data_root_path + xml_path + os.path.basename(image_file_name).split(".")[0] + ".xml"
            object_annotation = self._get_object_annotation(xml_file_name)
            self._draw_annotation(object_annotation, image_file_name)
        pass

    @staticmethod
    def _draw_annotation(object_annotation, image_file_name):
        im = Image.open(image_file_name).convert("RGB")
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("simhei.ttf", 15, encoding="utf-8")

        for object_one in object_annotation:
            object_type = CategoryNames[object_one[0]]
            x_min, y_min, x_max, y_max = object_one[1:]
            point = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]

            draw.line((point[0], point[1]), fill=(255, 0, 0), width=2)
            draw.line((point[1], point[3]), fill=(255, 0, 0), width=2)
            draw.line((point[3], point[2]), fill=(255, 0, 0), width=2)
            draw.line((point[2], point[0]), fill=(255, 0, 0), width=2)
            draw.text((point[0][0], point[0][1] - 20), object_type, (255, 0, 0), font=font)
            pass

        im.show(im)
        pass

    @staticmethod
    def _get_object_annotation(xml_file_name):
        reuslt = []
        tree = et.parse(xml_file_name)
        object_all = tree.getroot().findall("object")
        for object_one in object_all:
            object_type = CategoryNames.index(object_one.find("name").text)
            bndbox = object_one.find("bndbox")
            reuslt.append([object_type, int(bndbox.find("xmin").text), int(bndbox.find("ymin").text),
                           int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)])
            pass
        return reuslt

    pass

if __name__ == '__main__':
    # _data = Data(data_root_path="/home/z840/ALISURE/Data/VOC2012/",
    #              data_path="JPEGImages/", annotation_path="SegmentationObject/",
    #              data_list="ImageSets/Segmentation/train.txt")
    _data = Data(is_test=True)
    for i in range(20):
        final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_mask = _data.next_batch_train()
        _data.next_batch_test(i)
        pass
