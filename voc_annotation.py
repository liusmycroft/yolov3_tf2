import xml.etree.ElementTree as tr
import os

sets = ['train', 'val', 'test']
classes = ["grape", "apple", "banana", "orange", "watermelon"]


def convert_annotation(image_id, list_file):
    in_file = open('VOC_data/Annotations/%s.xml' % image_id)
    tree = tr.parse(in_file)
    root = tree.getroot()

    for obj in root.findall('object'):  # 创建以object为标签的迭代器
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        cls_id = classes.index(cls)     # 通过读取名字找到下标
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = os.getcwd()
for image_set in sets:
    image_ids = open('VOC_data/ImageSets/Main/%s.txt' %
                     image_set).read().strip().split()
    list_file = open('train_Set_Info/%s.txt' % image_set, 'w')
    for i in image_ids:
        list_file.write('%s/VOC_data/JPEGImages/%s.jpg' % (wd, i))
        convert_annotation(i, list_file)
        list_file.write('\n')
    list_file.close()
