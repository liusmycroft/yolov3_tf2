import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import colorsys
import os
from model import Yolo

class dectector(object):
    def __init__(self):
        super().__init__()
        self.model_path = 'logs/trained_weights.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/classes_path.txt'
        self.score = 0.3
        self.iou = 0.20
        self.model_image_size = (416,416)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def picture_process(self, image, size):
        iw, ih  = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)    # 某个方向扩大，某个方向缩小
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)   # 图片放大
        image.show()  
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        new_image.show()
        return new_image



    def detect_image(self, image):
        new_image = self.picture_process(image, self.model_image_size)
        print("=====预处理测试图片完成=====")
        image_data = np.array(new_image, dtype='float32')
        image_data /= 255.  # 归一化
        image_data = np.expand_dims(image_data, 0)  # 增加维度，给批次用
