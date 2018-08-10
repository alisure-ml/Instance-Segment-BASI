import os
import time
from skimage import io as sio
from pycocotools.coco import COCO
import numpy as np
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
                 batch_size=4, image_size=(720, 720), ratio=8, is_test=False, has_border=True):

        self.batch_size = batch_size
        self.image_size = image_size
        self.ratio = ratio
        self.num_segment = 4 if has_border else 3
        self.attention_class = 1
        self.num_classes = 21

        # 读取数据
        self._data_list, self._annotation_list, self._class_list = self._read_list(
            data_root_path, data_list, data_path, annotation_path, class_path)

        # test
        if is_test:
            self._data_list, self._annotation_list = self._data_list[0: 12], self._annotation_list[0: 12]
            self._class_list = self._class_list[0: 12]
            pass

        # 拆解标签
        self._annotations = self._read_annotation(self._annotation_list, self._class_list,
                                                  self.image_size, self.ratio, has_border)
        # 读取数据
        self._images_data = self._read_image(self._data_list, self.image_size)

        # 用来生成训练数据
        self.number_patch = len(self._annotations) // self.batch_size
        self._random_index = list(range(0, len(self._annotations)))
        self._now = 0
        pass

    @staticmethod
    def get_category_name(cat_id):
        return CategoryNames[cat_id]

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
    def _read_annotation(annotation_list, class_list, image_size, ratio, has_border=False):

        all_ann_data = []

        for ann_index, ann_name in enumerate(annotation_list):
            # 读取数据
            class_data = np.asarray(Image.open(class_list[ann_index]).resize(
                (image_size[0]//ratio, image_size[1]//ratio)))
            ann_data = np.asarray(Image.open(ann_name).resize((image_size[0]//ratio, image_size[1]//ratio)))

            # 边界当背景
            if has_border:
                ann_data = np.where(ann_data == 255, 171, ann_data)  # 2
            else:
                ann_data = np.where(ann_data == 255, 0, ann_data)
                pass

            # 所有的标注数字：其中0为背景，255为边界，其他为物体
            nums = [i for i in range(1, 255) if np.any(ann_data == i) and i != 171]

            # 所有标签的掩码和类别
            ann_mask = []
            ann_class = []
            for num in nums:
                # 掩码信息
                if has_border:
                    ann_mask_one = np.where(ann_data == num, 85, ann_data)  # 1
                    ann_mask_one = (ann_mask_one - 1) // 84
                else:
                    ann_mask_one = np.where(ann_data == num, 128, ann_data)  # 1
                    ann_mask_one = (ann_mask_one - 1) // 127
                    pass
                ann_mask.append(ann_mask_one)

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
    # 四种用途：图片名称+标注/图片数据+标注/图片名称/图片数据
    @staticmethod
    def load_image(image_filename_or_data_raw, where=None,
                   annotation_filename=None, ann_index=0, image_size=(720, 720)):

        # 读取数据
        if isinstance(image_filename_or_data_raw, str):
            data_raw = np.asarray(Image.open(image_filename_or_data_raw).resize(image_size), dtype=np.float32)
        else:
            data_raw = np.asarray(Image.fromarray(image_filename_or_data_raw).resize(image_size), dtype=np.float32)

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


class COCOData(object):

    def __init__(self, data_root_path="C:\\ALISURE\\DataModel\\Data\\COCO",
                 annotation_path="annotations_trainval2014\\annotations", data_type="val2014",
                 batch_size=4, image_size=(720, 720), ratio=8, min_area=400):

        self.batch_size = batch_size
        self.image_size = image_size
        self.ratio = ratio
        self.num_segment = 3
        self.attention_class = 2
        self.num_classes = 91

        # 读取数据
        self.img_path = os.path.join(data_root_path, data_type)
        self.ann_file = os.path.join(data_root_path, annotation_path, "instances_{}.json".format(data_type))

        self.coco = COCO(self.ann_file)

        self.ann_list = [{"image_id": self.coco.anns[ann_key]["image_id"],
                          "category_id": self.coco.anns[ann_key]["category_id"],
                          "id": self.coco.anns[ann_key]["id"],
                          "ann_ids": self.coco.getAnnIds(imgIds=self.coco.anns[ann_key]["image_id"]),
                          "file_name": os.path.join(
                              self.img_path,self.coco.loadImgs(ids=self.coco.anns[ann_key]["image_id"])[0]["file_name"])}
                         for ann_key in self.coco.anns.keys()
                         if self.coco.loadAnns(self.coco.anns[ann_key]["id"])[0]["area"] >= min_area]

        # 用来生成训练数据
        self.number_patch = len(self.ann_list) // self.batch_size
        self._random_index = list(range(0, len(self.ann_list)))
        self._now = 0

        print("---------------------------------------")
        print(self.number_patch)
        print(len(self.ann_list))
        print("---------------------------------------")

        pass

    @staticmethod
    def get_category_name(coco, cat_id):
        return coco.cats[cat_id]["name"]

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
        batch_ann = [self.ann_list[now_index] for now_index in now_indexes]

        batch_ann_data = []
        batch_ann_where = []
        batch_ann_mask = []

        # 选取初始点的位置
        _batch_ann_attention = []
        for ann_index, ann in enumerate(batch_ann):
            # attention
            anns_attention = self.coco.annToMask(self.coco.loadAnns(ann["id"])[0])
            _batch_ann_attention.append(anns_attention)
            batch_ann_data.append(np.zeros(anns_attention.shape, dtype=np.uint8))
            where = np.argwhere(anns_attention == 1)

            where = where[np.random.randint(0, len(where))]
            batch_ann_where.append(where)
            pass

        # ann 类别
        for ann_index, ann in enumerate(batch_ann):
            for ann_id in ann["ann_ids"]:
                load_anns = self.coco.loadAnns(ann_id)
                batch_ann_data[ann_index] += self.coco.annToMask(load_anns[0])
            # bg is 0, other is 1, attention is 2
            batch_ann_data[ann_index] = np.where(batch_ann_data[ann_index] > 0, 1, 0)
            batch_ann_data[ann_index] = np.where(_batch_ann_attention[ann_index] > 0, 2, batch_ann_data[ann_index])
            pass

        # 数据 and resize
        batch_data = []
        for ann_index, ann in enumerate(batch_ann):
            batch_data.append(np.asarray(Image.open(ann["file_name"]).convert("RGB").resize(self.image_size)) / 255)
            # batch_data.append(np.zeros(shape=(self.image_size[0], self.image_size[1], 3)))

            batch_ann_data[ann_index] = np.asarray(
                Image.fromarray(np.asarray(batch_ann_data[ann_index], dtype=np.uint8)).resize(
                    (self.image_size[0] // self.ratio, self.image_size[1] // self.ratio)))

            batch_ann_where[ann_index][0] = self.image_size[0] / len(_batch_ann_attention[ann_index]) * batch_ann_where[ann_index][0]
            batch_ann_where[ann_index][1] = self.image_size[1] / len(_batch_ann_attention[ann_index][0]) * batch_ann_where[ann_index][1]
            pass

        # 根据初始点生成高斯Mask
        for ann_index, _ in enumerate(batch_ann):
            batch_ann_mask.append(self._mask_gaussian(self.image_size, batch_ann_where[ann_index]))
            pass

        # 数据+MASK
        final_batch_data = []
        for one_data, one_mask in zip(batch_data, batch_ann_mask):
            final_batch_data.append(np.concatenate((one_data, np.expand_dims(one_mask, 2)), 2))
            pass

        # 标注
        final_batch_ann = [np.expand_dims(one_ann, 2) for one_ann in batch_ann_data]

        # 类别
        final_batch_class = [one_ann["category_id"] for one_ann in batch_ann]

        self._now += 1
        return final_batch_data, final_batch_ann, final_batch_class, batch_data, batch_ann_mask

    @staticmethod
    def _mask_gaussian(image_size, where, sigma=10):

        x = np.arange(0, image_size[1], 1, float)
        y = np.arange(0, image_size[0], 1, float)
        y = y[:, np.newaxis]

        x0 = where[1]
        y0 = where[0]

        # 生成高斯掩码
        mask = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(np.float32)
        return mask

    pass


if __name__ == '__main__':
    # _data = Data(is_test=True)
    # for i in range(20):
    #     _data.next_batch_train()
    #     _data.next_batch_test(i)
    #     pass

    data_reader = COCOData(data_root_path="/home/z840/ALISURE/Data/COCO",
                           annotation_path="annotations_trainval2014/annotations",
                           data_type="train2014", batch_size=2, image_size=[720, 720])
    for i in range(200):
        data_reader.next_batch_train()
        pass

