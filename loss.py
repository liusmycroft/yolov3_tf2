import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.backend import switch as sw
from math import isnan


class yolo_Loss(tf.keras.losses.Loss):
    def __init__(self, anchors, num_classes, ignore_thresh=.5, input_shape=(416,416)):
        super().__init__()
        print()
        print("loss函数初始化！")
        print()
        self.thresh = ignore_thresh
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_layers = len(anchors) // 3
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = input_shape
        
    def call(self, y_true, y_pred):
        # y_true的数据是经过归一化的，相对于整个图像的比例
        tf.compat.v1.enable_eager_execution()
        for i in range(3):
            y_true[i] = tf.convert_to_tensor(y_true[i], dtype=tf.float32)
            y_pred[i] = tf.convert_to_tensor(y_pred[i], dtype=tf.float32)
        
        m = K.shape(y_true[0])[0]
        m_float = K.cast(m, K.dtype(y_true[0]))
        # 初始化loss
        loss = 0

        # 创建规格图，给y_true用
        grid_shapes = [
            K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0]))
            for l in range(self.num_layers)
        ]
        for i in range(self.num_layers):
            
            # 对预测结果处理，返回的是相对于规格图(13,13)的比例
            # box_xy, box_wh, box_confidence, box_class_probs, grid = prediction_result_process(y_pred[i], self.anchors[self.anchor_mask[i]], self.num_classes, self.input_shape)
            box_xy, box_wh, grid, feats = prediction_result_process(y_pred[i], self.anchors[self.anchor_mask[i]], self.num_classes, self.input_shape, flag_cal_loss=True)
            # 将xy和wh组合成预测框pred_box，结构是(batch_size, 13, 13, 3, 4)
            pred_box = K.concatenate([box_xy, box_wh])
            
            object_mask = y_true[i][..., 4:5]
            object_mask_boolen = K.cast(object_mask, 'bool')
            true_class_probs = y_true[i][..., 5:]

            # 计算出真实值的偏移程度，与预测值一致
            true_xy = y_true[i][..., :2] * grid_shapes[i][::-1] - grid 
            # 将真实值倒推回去
            true_wh = K.log(y_true[i][..., 2:4] / self.anchors[self.anchor_mask[i]] * self.input_shape[::-1])
            true_wh = sw(object_mask, true_wh, tf.zeros_like(true_wh))
            # np.set_printoptions(threshold=np.inf)
            # file = open('data.txt','w')
            # file.write(str(y_true))
            # file.close()
            # exit()
            box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            for j in range(m):
                true_box = tf.boolean_mask(y_true[i][j, ..., 0:4], object_mask_boolen[j, ..., 0])
                iou = box_iou(pred_box[j], true_box)
                best_iou = K.max(iou, axis=-1)
                # 将(13, 13, 3)的信息按批次下标写入列表
                ignore_mask = ignore_mask.write(j, K.cast(best_iou < self.thresh, K.dtype(true_box)))
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)
            
            # test1 = K.sum(true_wh) / m_float
            # test2 = K.sum(feats[..., 2:4]) / m_float
            # test3 = K.sum(true_xy) / m_float
            # test4 = K.sum(feats[..., 0:2]) / m_float
            # print("===============================")
            # print("规格类型为：", i+1)
            # print("true_wh: ", test1.numpy())
            # print("box_wh: ", test2.numpy())
            # print("true_xy: ", test3.numpy())
            # print("box_xy: ", test4.numpy())
            # print("===============================")

            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(true_xy, feats[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(true_wh - feats[..., 2:4])
            # 后面那个部分是“没有物体”，但是有可能物体在附近，看iou来决定是否要算入误差内
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, feats[..., 4:5], from_logits=True) + (1-object_mask) * K.binary_crossentropy(object_mask, feats[..., 4:5], from_logits=True) * ignore_mask
            class_loss = object_mask * K.binary_crossentropy(true_class_probs, feats[..., 5:], from_logits=True)
            
            xy_loss = K.sum(xy_loss) / m_float
            wh_loss = K.sum(wh_loss) / m_float
            confidence_loss = K.sum(confidence_loss) / m_float
            class_loss = K.sum(class_loss) / m_float
            
            # print("===============================")
            # print("规格类型为：", i+1)
            # print("xy_loss: ", xy_loss.numpy())
            # print("wh_loss: ", wh_loss.numpy())
            # print("confidence_loss: ", confidence_loss.numpy())
            # print("class_loss: ", class_loss.numpy())
            # print("===============================")
            # print()
            if(isnan(xy_loss.numpy())):
                print("xy_loss 爆了")
            if(isnan(wh_loss.numpy())):
                print("wh_loss 爆了")
            if(isnan(confidence_loss.numpy())):
                print("confidence_loss 爆了")
            if(isnan(class_loss.numpy())):
                print("class_loss 爆了")
            loss += xy_loss + wh_loss + confidence_loss + class_loss
        return loss


def box_iou(b1, b2):

    # Expand dim to apply broadcasting.
    # shape=(13, 13, 3, 1, 5 + classes)
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    # shape=(13, 13, 3, 1, 2)
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    # shape=(1, 5, 5 + classes) ---> （在某种规格下）这张图有5个GD框
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    # shape=(1, 5, 2)
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # shape=(13, 13, 3, 1, 2) + shape=(1, 5, 2)
    # shape=(13, 13, 3, 5, 2)
    # 相当于先用每个网格里面的某种先验框的预测框信息与多个GD框相比，再遍历3种先验框
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# 返回的是根据模型的输出而做出处理的xywh，模型输出的是"偏置值"，需要转化成具体长度和位置，再做归一化！！
def prediction_result_process(feats, anchors, num_classes, input_shape, flag_cal_loss):
    input_shape = np.array(input_shape, dtype='int32')
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    
    # 生成网格背景
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    
    # 预测数据预处理
    # (2, 13, 13, 90) --> (6, 13, 13, 3, 10)
    # 将原本混在一起的先验框分开
    feats = K.reshape(
        feats,
        [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
   
    # 开始转换，将背景网格与预测结果相加并归一化
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(
        grid_shape[::-1], K.dtype(feats))
    box_wh = (K.exp(feats[..., 2:4]) * anchors_tensor) / K.cast(
        grid_shape[::-1], K.dtype(feats))

    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
    if(flag_cal_loss):
        return  box_xy, box_wh, grid, feats
    return box_xy, box_wh, box_confidence, box_class_probs, grid
