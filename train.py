import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from public import get_classes,get_anchors,data_generator
from model import Yolo
from loss import yolo_Loss
import datetime as dt
from math import isnan
import tqdm


def create_native_model(anchors_nums, num_classes):
    yolo = Yolo(num_anchors=anchors_nums, num_classes=num_classes)
    return yolo

def train(model,
          annotation_path,
          input_shape,
          anchors,
          num_classes,
          log_dir='logs/',
          epoch=1,
          batch_size=3):
    # 定义损失函数
    yolo_loss_func = yolo_Loss(anchors=anchors,num_classes=num_classes,ignore_thresh=0.5, input_shape=input_shape)
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    # 定义训练集占总样本的比例
    val_split = 0.1
    # 打开图片路径，打乱训练样本
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    # 计算一次遍历所有要多少个 batch_size
    times_in_one_epoch = int(len(lines) / batch_size)
    # times_in_one_epoch = 20
    # 计算训练集/验证集的个数
    num_val = int(len(lines) * val_split)   # 验证集个数
    num_train = len(lines) - num_val        # 训练集个数
    print("=====================================================")
    print("=  准被就绪，开始训练，本次训练的样本数为：%d" % num_train)
    print("=  训练批次大小为：%d" % batch_size)
    print("=  总共训练次数为：%d" % epoch)
    print("=  times_in_one_epoch：%d" % times_in_one_epoch)
    print("=====================================================")
    # 定义迭代器，取数据用
    tem = data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    
    # 核心，开始训练
    for j in range(epoch):
        starttime = dt.datetime.now()
        total_loss = 0
        for i in tqdm.trange(times_in_one_epoch, ncols=75):     # 遍历一次则跑遍所有样本
            abolo = next(tem)   # 当 batch_size=2时，abolo大小为[2, ]，abolo[0]大小为[2,416,416,3]，abolo[1][0]大小为[2,13,13,3,10]
            with tf.GradientTape() as tape:
                res = model(abolo[0])
                loss = yolo_loss_func(abolo[1], res)
                #print("分为%d批次,这次是第%d次批次," % (times_in_one_epoch, (i+1)), end = '')
                #print("损失值为：", loss.numpy())
                if(isnan(loss.numpy())):
                    print("他妈的爆表了")
                    exit()
                total_loss += loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            
        endtime = dt.datetime.now()
        print()
        print("=================================================")
        print("= 当前训练批次为：%d" % j)
        print("= 本次epoah耗时为:", (endtime - starttime).seconds,"s", "= 本次epoach平均损失为:", total_loss.numpy()/times_in_one_epoch)
        print("=================================================")
        print()
    
    
    model.save_weights(log_dir + 'trained_weights.h5')


def _main():
    epochs = 1
    annotation_Path = 'train_Set_Info/train.txt'
    log_Dir = 'logs/'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_size = 5
    batch_size = 2
    anchors = get_anchors(anchors_path)  # 获取anchor_box
    anchors_size = 3
    input_shape = (416, 416)  # multiple of 32, hw
    model = create_native_model(anchors_size, class_size)
    train(model,
          annotation_Path,
          input_shape,
          anchors,
          class_size,
          log_dir=log_Dir,
          epoch=epochs,
          batch_size=batch_size)

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    _main()
    