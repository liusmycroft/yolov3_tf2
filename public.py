import numpy as np
from PIL import Image
import tensorflow as tf


def get_classes(classes_path):
    with open(classes_path) as f:
        name = f.readlines()
    name = [c.strip() for c in name]
    return name


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors,
                   num_classes):
    n = len(annotation_lines)  # 训练图片的路径文件的总数，有多少条
    if n == 0 or batch_size <= 0:
        print("n==0 OR batch_size == 0 ! Please check it!")
        return None
    np.random.shuffle(annotation_lines)  # 打乱训练集
    i = 0
    while True:
        image_data = []
        box_data = []
        # 开始生成一个batch_size的训练集
        # i：从打乱的训练集中抽取的数据的下标，会递增，若增大到所有数据的个数，则重头训练
        for b in range(batch_size):
            i %= n
            image, box = get_Preprocess(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors,
                                       num_classes)
        # 生成 image数组和对应的 y_true，generator参数要求返回一个元组 [inputs, target]
        # y_true = tf.Variable(y_true, tf.float32)
        yield image_data, y_true  # 下次跳到 while处执行


# preprocess_true_boxes为上面的 data_generator生成 train样本对应 y_true
# true_boxes为未加工的，但是做了放大后的处理
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 顺序颠倒，小的先验框对应小的视野范围，即对应52*52
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # (x_min + x_max) // 2 = middle，已经完成所有数据（一个batch）的处理
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 此时true_boxes存的不再是 x_max 等，而是归一化后的xywh，除(416,416)注意要与预测值做相同的处理！！
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]  # 批次！！

    # 将生成 (13*13) (26*26) (52*52) 的模板
    # [[13,13],[26,26],[52,52]]
    grid_shapes = [
        input_shape // {
            0: 32,
            1: 16,
            2: 8
        }[l] for l in range(num_layers)
    ]
    # 构造y_true(模型输出的东西)的模板，有三种
    # 注意前面的 m，为批次大小
    # 形状为 [规格数量(3种规格),批次大小,规格下的长，规格下的宽，规格下的先验框种类数(3种先验框)，xywh + classes]
    y_true = [
        np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(
            anchor_mask[l]), 5 + num_classes),
                 dtype='float32') for l in range(num_layers)
    ]

    # 此时 y_true的形状为 [批次大小,一张图片中盒子的个数,盒子的信息(x_min,y_min,x_max,y_max)]
    # 而 anchors的形状为 [先验框的个数,先验框的信息（长/宽）]
    # 要把两个放在一起算的话要给 anchors扩充一维
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.  # 如(10，13)会变成 (5，6.5)
    anchor_mins = -anchor_maxes
    # 读取一个batch中所有的宽的信息（打竖地读）
    # valid_mask 完成后是以批次划分的，形状为 [批次大小，有一张图片中盒子的个数多的True/False掩码]
    valid_mask = boxes_wh[..., 0] > 0

    # 循环针对的一个 batch中的某个训练样品
    for b in range(m):
        # 此时 boxes_wh的形状为 [批次大小,一张图片中盒子的个数,盒子的宽/高]
        wh = boxes_wh[b, valid_mask[b]]  # width小于0直接把整个部分清除，变成[]
        if len(wh) == 0:
            continue
        # 剔除 width < 0的盒子完毕
        # wh的形状为 [一张图片中剔除完毕后盒子的个数，盒子的宽/高信息]
        wh = np.expand_dims(
            wh, -2
        )  # 在最后一个维度前加一层，即[[30, 31],[32, 33]] --> [ [[30, 31]], [[32, 33]] ]

        box_maxes = wh / 2.
        box_mins = -box_maxes  # 假设原来的 w，h=(20, 6),现在 box_mins变成 (-10, -3)

        # 一个 y_true框输进去，将会与多个(9)先验框进行比较，分别输出与先验框比较大小的结果，如：一个 y_true框对应 9个结果
        intersect_mins = np.maximum(
            box_mins,
            anchor_mins)  # 比较(-5,-6.5) 与 (-10, -3)返回哪个最大，结果为 (-5, -3)
        intersect_maxes = np.minimum(
            box_maxes, anchor_maxes)  # 比较(5,6.5) 与 (10, 3)返回哪个最小，结果为 (5, 3)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]  # y_true框本来的大小 注意...是纵轴
        anchor_area = anchors[..., 0] * anchors[..., 1]  # 先验框大小，共九个面积
        # 计算 iou，交叉部分在十字架中的占比，shape形如：[[0, 2, 4],[1, 2, 8]]
        # [[x],[y]] 中x代表了第一个 y_true框与9种先验框的 iou值
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(
            iou, axis=-1
        )  # 选择与 y_true框重合度最高的那个的下标，形如：(2,2) --> (4, 8) ----> 对应的上面的 [[0, 2, 4],[1, 2, 8]]

        for t, n in enumerate(best_anchor):  # n = 6 ---> 对应第七种先验框，意思是第一个 y_true框与第七种先验框的 IOU最大
            for l in range(num_layers):  # 设 l = 2
                if n in anchor_mask[l]:  # anchor_mask[0] = [6,7,8] ---> 对应最后三种先验框(7，8，9)
                    # true_boxes[b, t, 0] ---> 代表 batch中的第 b张图片中的第 t个 ground_truth框的中心点的归一化的 x轴坐标
                    # grid_shapes指要填入何种规格的框，有 13*13/52*52/26*26
                    i = np.floor(true_boxes[b, t, 0] *
                                 grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] *
                                 grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)  # 查看这个 y_true框最终使用的是哪种先验框
                    c = true_boxes[b, t, 4].astype('int32')
                    # 针对第 k个先验框的柜子装入 y_true信息，其余未被选中的框的柜子内统统为 0
                    # k = {0,1,2}
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1  # 置信度标记为 1，确实有物体存在
                    y_true[l][b, j, i, k, 5 + c] = 1  # 对应类别标记为 1
        # 这意味着不同的 y_true框会根据自身与先验框的重合情况被分配到不同的规格图中
        # 比如小的物体与前三种先验框iou大
        # 最后的 y_true呈现出三种规格，同时存有每种规格图的每个格子的偏移量
    return y_true


def get_Preprocess(annotation_line, input_shape, max_boxes=15):
    line = annotation_line.split()
    image = Image.open(line[0])
    # image.show()
    iw, ih = image.size  # 输入图片的宽/高
    h, w = input_shape  # 指定的input的宽/高
    box = np.array(
        [np.array(list(map(int, box.split(',')))) for box in line[1:]])

    scale = min(w / iw, h / ih)  # 转换的最小比例
    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)  # 扩大/缩小到新的尺寸（就是指定的input尺寸）
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image_data = 0
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))  # 生成灰色图像
    new_image.paste(image, (dx, dy))  # (left, upper),距离左边的距离和距离上边的距离
    image_data = np.array(new_image) / 255.  # 归一化
    box_data = np.zeros((max_boxes, 5))  # x_min, y_min, x_max, y_max, class_id

    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]  # 最多只能有20个对象
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx  # 按比例缩放并加上偏置值
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box_data[:len(box)] = box
    else:
        print("******* length of box == 0! *********")

    return image_data, box_data
