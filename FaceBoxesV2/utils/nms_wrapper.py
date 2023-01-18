# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# import sys
# sys.path.append('D:\\13219\\Desktop\\PIPNet\\FaceBoxesV2\\utils')
# # print(sys.path)
# from nms.cpu_nms import cpu_nms#, cpu_soft_nms
# # print(os.getcwd())

# sys.path.insert(0, 'FaceBoxesV2')
# # sys.path.append('D:\\13219\\Desktop\\PIPNet\\FaceBoxesV2\\utils')
# # print(sys.path)
# from FaceBoxesV2.utils.nms.cpu_nms import cpu_nms#, cpu_soft_nms

# from .nms.cpu_nms import cpu_nms, cpu_soft_nms

import numpy as np

def mymax(a, b):
    return a if a >= b else b

def mymin(a, b):
    return a if a <= b else b

def cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = mymax(ix1, x1[j])
            yy1 = mymax(iy1, y1[j])
            xx2 = mymin(ix2, x2[j])
            yy2 = mymin(iy2, y2[j])
            w = mymax(0.0, xx2 - xx1 + 1)
            h = mymax(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep




def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets, thresh)

# Not Needed
# 我让这里的import不报错后，再生成的exe找不到cv了……
# 不过尚不清楚是哪个文件的import找不到它可爱的cv了
