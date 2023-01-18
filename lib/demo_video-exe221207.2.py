# ==Switch==
contour_thin = 1  # tag: 缩脸
contour_mapping = 0  # tag: 映射mapping。abandon 废弃
schema_mask = 1.1  # tag: 曲线轮廓bezier curve
smooth_vertical = 0  # 深度可变
smooth_parallel = 0  # 固定距离时多组参数
direct_mapping = 1  # 直接映射：从Im映射到Mi，不经过3D坐标Ca_coor
# ==/Switch==

import sys
sys.path.insert(0, '../')
from tools.copicks import *

from distutils.log import error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json

from tkinter.messagebox import NO, YES
import cv2, os
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
sys.path.insert(0, r'..\FaceBoxesV2')
sys.path.insert(0, r'...')
import numpy as np
# import pickle
# import importlib
# from math import floor
# from faceboxes_detector import *
from FaceBoxesV2.faceboxes_detector import *
# import time

import pyrealsense2 as rs
import cv2
# import csv
# import pandas as pd

import torch
# import torch.nn as nn
import torch.nn.parallel
# import torch.optim as optim
import torch.utils.data
# import torch.nn.functional as F
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
# import data_utils
from functions import *

# python lib/demo_video.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera
# python lib/demo_video_test_stable.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera

# if not len(sys.argv) == 3:
#     print('Format:')
#     print('python lib/demo_video.py config_file video_file')
#     exit(0)
# experiment_name = sys.argv[1].split('/')[-1][:-3]  # pip_32_16_60_r18_l2_l1_10_1_nb10
# data_name = sys.argv[1].split('/')[-2]  # WFLW
# config_path = '.experiments.{}.{}'.format(data_name, experiment_name)  # .experiments.WFLW.pip_32_16_60_r18_l2_l1_10_1_nb10
# video_file = sys.argv[2]  # camera

# TODO 项目代码放入GitHub中，方便迭代。

experiment_name = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
data_name = 'WFLW'
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)  # .experiments.WFLW.pip_32_16_60_r18_l2_l1_10_1_nb10
video_file = 'camera'


# print(config_path)
# my_config = importlib.import_module(config_path, package='PIPNet')
# my_config = r''



# Config = getattr(my_config, 'Config')

class Config():
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.input_size = 256
        self.backbone = 'resnet18'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 98
        self.save_interval = self.num_epochs
        self.num_nb = 10
        self.use_gpu = True
        self.gpu_id = 2


cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

save_dir = os.path.join('../snapshots', cfg.data_name, cfg.experiment_name)
# save_dir = r'../snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth'
# print(save_dir)

meanface_file = r'../data/WFLW/meanface.txt'
meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(meanface_file, cfg.num_nb)
# meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

if cfg.backbone == 'resnet18':
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet50':
    resnet50 = models.resnet50(pretrained=cfg.pretrained)
    net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet101':
    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
else:
    print('No such backbone!')
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)


weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

###transpose 2022.9.6
#### mi_x = px(ca_x)
# co_x = [-4323.70664676, 1041.6113063]  # abandon
# co_x = [-4319.05267295, 1053.73707038]
## 
#### mi_y = np.dot([1,ca_y,ca_z],co_y)
# co_y = [218.9787, 4133.2564, 1197.3876]  # abandon
# co_y = [-206.83741549, 4117.60761919, 1080.1952626]
# co_y = np.array(co_y)
##

# co_x = [-4624.19868258, 986.23460526]  # 距离50cm
# co_y = np.array([-47.91786684, 4183.69022911, 1011.09430934])  # 距离50cm
# co_x = [-4977.67274306, 1196.21840906]  # eye_right
# co_y = np.array([ 227.84231818, 3831.45072206, 164.93831212])  # eye_right
# co_x = [-4452.53363713, 896.03404077]  # eye_left
# co_y = np.array([-676.25001287, 5444.76764539, 1168.85784709])  # eye_left

# co_30l = np.loadtxt("../data/mapping/30l.txt", delimiter=",", dtype=float)
# co_40l = np.loadtxt("../data/mapping/40l.txt", delimiter=",", dtype=float)
# co_50l = np.loadtxt("../data/mapping/50l.txt", delimiter=",", dtype=float)
# co_60l = np.loadtxt("../data/mapping/60l.txt", delimiter=",", dtype=float)

def demo_video(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', '../FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.9
    det_box_scale = 1.2
    init_Ca_coor = 1
    init_Mi_coor = 1

    if contour_mapping == 1:  # 映射mapping
        co_mapping = json.load(open('./trans.json', 'r'))
        co_mapping = np.array(co_mapping)

    net.eval()

    #### start realsense 2022.9.6
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    frame_height = 1280
    frame_width = 720
    # frame_height = 640
    # frame_width = 480
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流
    
    # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, frame_height, frame_width, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, frame_height, frame_width, rs.format.bgr8, 30)
    
    pipe_profile = pipeline.start(config)  # streaming流开始
    
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐

    # ==曲线轮廓函数bezier curve==
    def get_bezier_coef(points):
        # since the formulas work given that we have n+1 points
        # then n must be this:
        n = len(points) - 1

        # build coefficents matrix
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2

        # build points vector
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]

        # solve system, find a & b
        A = np.linalg.solve(C, P)
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        return A, B

    # returns the general Bezier cubic formula given 4 control points
    def get_cubic(a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

    # return one cubic curve for each consecutive points
    def get_bezier_cubic(points):
        A, B = get_bezier_coef(points)
        return [
            get_cubic(points[i], A[i], B[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    # evalute each cubic curve on the range [0, 1] sliced in n points
    def evaluate_bezier(points, n):
        curves = get_bezier_cubic(points)
        # print(np.array(curves).shape)
        # print(curves)
        return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])
    # ==/曲线轮廓函数bezier curve==

    csv_flag = 0  # 用于保存csv文件

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
            #### 获取相机参数 ####
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

            if not aligned_depth_frame or not aligned_color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
 
            color_image = np.asanyarray(aligned_color_frame.get_data())

            # depth_image = cv2.flip(depth_image, 1)
            # color_image = cv2.flip(color_image, 1)
 
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            frame = color_image
            # Show images
            
            # print("shape is", frame.shape)
            # print("0度是", frame.shape[0])
            # print("1度是", frame.shape[1])
            # break

            def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin, depth_base=0):
                x = depth_pixel[0]
                y = depth_pixel[1]
                try:
                    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
                except RuntimeError as err:  # https://github.com/IntelRealSense/librealsense/issues/7395
                    # print(err)
                    dis = 0
                    pass
                # dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
                # print ('depth: ',dis)       # 深度单位是m
                # DONE 需要一个dep_base
                if depth_base!=0 and (dis==0 or dis>1.0):
                    dis = depth_base
                camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
                # print ('camera_coordinate: ',camera_coordinate)
                return dis, camera_coordinate
            # def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
            #     x = depth_pixel[0]
            #     y = depth_pixel[1]
            #     dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
            #     # print ('depth: ',dis)       # 深度单位是m
            #     camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
            #     # print ('camera_coordinate: ',camera_coordinate)
            #     return dis, camera_coordinate

            def on_EVENT_BUTTONDOWN(event, x, y,flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # print("1:",images[y][x][0])
                    # print("2:",images[y][x][1])
                    # print("3:",images[y][x][2])
                    print("camera_click")

                    depth, ca_coor = get_3d_camera_coordinate([x,y],aligned_depth_frame,depth_intrin)
                    # print(depth, ca_coor)
            
                    
                    # temp = [x,y,*ca_coor]
                    [Ca_x, Ca_y, Ca_z] = [*ca_coor]
                    Mi_x = px(Ca_x)
                    y = np.array([1, Ca_y, Ca_z])
                    # print(y.shape,co_y.shape)
                    Mi_y = np.dot(y,co_y)
                    xy = "%d,%d" % (int(Mi_x),int(Mi_y))

                    # re.append(temp)
                    # im_x.append(x)
                    # im_y.append(y)
                    # ca_x.append(a)
                    # ca_y.append(b)
                    # ca_z.append(c)

            
                    cv2.circle(Mi_p, (int(Mi_x), int(Mi_y)), 10, (255, 0, 0), thickness=-1)
                    cv2.putText(Mi_p, xy, (int(Mi_x), int(Mi_y)), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0), thickness=1)
                    cv2.imshow("Mi", Mi_p)

                    # result.append(xy)
            # def on_EVENT_BUTTONDOWN2(event, x, y, flags, param):        
            #     if event == cv2.EVENT_LBUTTONDOWN:
            #         print("mirror_click")
            #         # mi_x.append(x)
            #         # mi_y.append(y)



            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)#WINDOW_NORMAL
            # Mp_height = 1200
            # Mp_width = 1750
            Mp_height = 1920#1920+10
            Mp_width = 1200#1200+20
            Mi_p = np.zeros((Mp_height, Mp_width,3), np.uint8) 
            # cv2.namedWindow('Mirror_P', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Mirror_P',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Mirror_P', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            frame_width = 720
            frame_height = 1280

            ### detect
            detections, _ = detector.detect(frame, my_thresh, 1)
            # detections, _ = detector.detect(image=frame, thresh=my_thresh)#, 1)
            # print(len(detections))
            flag_text = 1
            for i in range(len(detections)):  # detections存储人脸检测目标框，len(detections)对应人脸数量
                det_xmin = detections[i][2]
                det_ymin = detections[i][3]
                det_width = detections[i][4]
                det_height = detections[i][5]
                det_xmax = det_xmin + det_width - 1
                det_ymax = det_ymin + det_height - 1

                # XXX 检测框可能有问题
                # det_xmin -= int(det_width * (det_box_scale-1)/2)
                # # remove a part of top area for alignment, see paper for details
                # det_ymin += int(det_height * (det_box_scale-1)/2)
                # det_xmax += int(det_width * (det_box_scale-1)/2)
                # det_ymax += int(det_height * (det_box_scale-1)/2)
                # det_xmin = max(det_xmin, 0)
                # det_ymin = max(det_ymin, 0)
                # det_xmax = min(det_xmax, frame_width-1)
                # det_ymax = min(det_ymax, frame_height-1)
                # det_width = det_xmax - det_xmin + 1
                # det_height = det_ymax - det_ymin + 1

                # cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
                # DONE 提高曲线效果
                # 1. 调整分辨率，Config里的默认分辨率似乎是256 CANCEL
                # 2.1. 缩脸 DONE
                # 2.2. 映射 CANCEL
                # print(np.array(det_crop).shape)
                if np.array(det_crop).shape[0]*np.array(det_crop).shape[1] == 0:
                    # print('donedonedone')
                    continue
                det_crop = cv2.resize(det_crop, (input_size, input_size))
                inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
                inputs = preprocess(inputs).unsqueeze(0)
                inputs = inputs.to(device)
                lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
                lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
                tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
                lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                lms_pred = lms_pred.cpu().numpy()
                lms_pred_merge = lms_pred_merge.cpu().numpy()
# DONE 优化points
# 1. 集中存储point位置，避免重复计算 DONE
# 1.1. 移出循环，放入字典里 DONE
# 2. 分析point的z轴数据，平滑显示 DONE
                # print(lms_pred_merge)
                # print(lms_pred_merge.shape)
                # print(type(lms_pred_merge))
                # x_pred = []
                # y_pred = []
                x_pred = np.array([])
                y_pred = np.array([])
                for i in range(98):
                    # x_pred.append(int(lms_pred_merge[i*2] * det_width))
                    # y_pred.append(int(lms_pred_merge[i*2+1] * det_height))
                    x_pred = np.append(x_pred, lms_pred_merge[i*2] * det_width)
                    y_pred = np.append(y_pred, lms_pred_merge[i*2+1] * det_height)
                
                Im_x = x_pred + det_xmin
                Im_y = y_pred + det_ymin
                Im_x = Im_x.astype(int)
                Im_y = Im_y.astype(int)

                # ==映射mapping==
                if contour_mapping == 1:
                    ipt_x = Im_x[33:]
                    ipt_y = Im_y[33:]
                    # mapping_ipt = np.vstack([np.ones(len(ipt_x)).astype('int'), ipt_x, ipt_y]).T.reshape(-1, 1)
                    # mapping_ipt = np.vstack([np.ones(len(ipt_x)).astype('int'), ipt_x, ipt_y]).reshape(-1, 1)
                    mapping_ipt = np.array([
                        np.ones(len(ipt_x)).astype('int'),
                        ipt_x,
                        ipt_y
                    ]).reshape(-1, 1)
                    mapping_opt_x = np.dot(co_mapping[:, :, 0], mapping_ipt).reshape(-1)
                    mapping_opt_y = np.dot(co_mapping[:, :, 1], mapping_ipt).reshape(-1)
                    # print('x:', mapping_opt_x.shape)
                    # print('y:', mapping_opt_y.shape)
                    # print('x:', mapping_opt_x)
                    # print('y:', mapping_opt_y)
                    # break
                    # exit()
                    Im_x[:33] = mapping_opt_x
                    Im_y[:33] = mapping_opt_y
                # ==/映射mapping==

                # csv_flag += 1
                # if csv_flag%100 == 0:
                #     csv_num = int(csv_flag/100)
                #     df_csv = pd.DataFrame({
                #         'Im_x': Im_x,
                #         'Im_y': Im_y
                #     })
                #     df_csv.to_csv('coord_Im/coord_Im{}.csv'.format(csv_num), index=True)

                # 曲线轮廓bezier curve。取Im_points_base需要在缩脸之前
                Im_points_base = np.array([[Im_x[i], Im_y[i]] for i in range(len(Im_x))])

                # ==缩脸 - Im缩==
                if contour_thin == 1:
                    co_thin = 5
                    for i in range(16):
                        Im_x[i] += co_thin
                        Im_y[i] -= co_thin
                    Im_y[16] -= co_thin
                    for i in range(16, 33):
                        Im_x[i] -= co_thin
                        Im_y[i] -= co_thin
                # ==/缩脸 - Im缩==

                def fun_depth_base(Im_x, Im_y, aligned_depth_frame, depth_intrin):
                    """
                        设置人脸基准深度
                        当关键点深度检测值异常时，用基准深度代替
                    """
                    # # 左右眼深度求平均
                    # depth_base1, _= get_3d_camera_coordinate([Im_x[96], Im_y[96]], aligned_depth_frame, depth_intrin)
                    # depth_base2, _= get_3d_camera_coordinate([Im_x[97], Im_y[97]], aligned_depth_frame, depth_intrin)
                    # depth_base = (depth_base1 + depth_base2) / 2

                    # 人中深度
                    depth_base, _ = get_3d_camera_coordinate([Im_x[79], Im_y[79]], aligned_depth_frame, depth_intrin)
                    return depth_base
                depth_base = fun_depth_base(Im_x, Im_y, aligned_depth_frame, depth_intrin)
                if flag_text == 1:
                    depth_text = depth_base
                    flag_text = 0

                # 求外轮廓平均值
                depth_outline = np.array([])
                for i in range(33):
                    dep, _ = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin)
                    if dep>0 and dep<1.0:
                        depth_outline = np.append(depth_outline, dep)
                depth_outline = np.mean(depth_outline)

                if init_Ca_coor == 0:
                    for i in range(98):  # 98个关键点，逐个获取其在相机坐标系中的坐标
                        # 深度值异常时，以关键点97的深度值代替 CANCEL
                        dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin)
                        # if dep > 1.5:
                        #     print(cc)
                        if dep == 0 or dep > 1.0:
                            # # 粗略分组，抖动较弱
                            # if i in range(55):
                            #     for _ in range(15):
                            #         i_bias = np.random.randint(1, 16)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i+i_bias], Im_y[i+i_bias]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # # if i in range(33):  # 新思路，但效果差
                            # #     # for _ in range(30):
                            # #     # for i_rand in np.random.randint(0, 33, 30):
                            # #     for i_rand in np.random.randint(55, 98, 30):
                            # #         # i_bias = np.random.randint(1, 16)
                            # #         # i_rand = np.random.randint(33)
                            # #         # dep_neigh, _ = get_3d_camera_coordinate([Im_x[i+i_bias], Im_y[i+i_bias]], aligned_depth_frame, depth_intrin)
                            # #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            # #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            # #         if dep!=0 and dep<1.0:
                            # #             break
                            # else:
                            #     i_rand = np.random.randint(55, 98)
                            #     dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #     dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)

                            ## 更细致的分组，抖动较大
                            # if i in range(16):  # 侧脸
                            #     for _ in range(15):
                            #         i_rand = np.random.randint(1, 16)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # elif i in range(16, 33):  # 侧脸
                            #     for _ in range(15):
                            #         i_rand = np.random.randint(16, 33)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # elif i in range(33, 42):  # 眉毛
                            #     for _ in range(5):
                            #         i_rand = np.random.randint(33, 42)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # elif i in range(42, 51):  # 眉毛
                            #     for _ in range(5):
                            #         i_rand = np.random.randint(42, 51)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # elif i in range(51, 55):  # 鼻子
                            #     for _ in range(3):
                            #         i_rand = np.random.randint(51, 55)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # else:
                            #     i_rand = np.random.randint(55, 98)
                            #     dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #     dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)

                            # # 更科学的分组，抖动一般
                            # if i in range(33, 55):
                            #     # for _ in range(15):
                            #     for i_rand in np.random.randint(33, 55, 15):
                            #         # i_rand = np.random.randint(33, 55)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # elif i in range(55, 98):
                            #     # for _ in range(15):
                            #     for i_rand in np.random.randint(55, 98, 15):
                            #         # i_rand = np.random.randint(55, 98)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            # else:  # i in range(33):
                            #     # for _ in range(30):
                            #     for i_rand in np.random.randint(0, 33, 30):
                            #         # i_rand = np.random.randint(33)
                            #         dep_neigh, _ = get_3d_camera_coordinate([Im_x[i_rand], Im_y[i_rand]], aligned_depth_frame, depth_intrin)
                            #         dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, dep_neigh)
                            #         if dep!=0 and dep<1.0:
                            #             break
                            
                            # 外轮廓平均值的方法，抖动一般
                            if i in range(33):
                                dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, depth_outline)

                        # dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, depth_base)

                        if dep == 0 or dep > 1.0:  # 异常值用上一帧数据代替
                            continue
                        Ca_coor[i] = cc
                else:  # 初始化变换后的坐标值，只运行一次
                    Ca_coor = np.array([])
                    for i in range(98):
                        dep, cc = get_3d_camera_coordinate([Im_x[i], Im_y[i]], aligned_depth_frame, depth_intrin, depth_base)
                        Ca_coor = np.append(Ca_coor, cc)
                        # if dep == 0 or dep > 1.5:
                        #     print(i, cc)
                    Ca_coor = Ca_coor.reshape(-1, 3)
                    # print(Ca_coor)
                    init_Ca_coor = 0

                # DONE 随深度改变mask大小
                # 1. 取dep的十数位，传入判断case的函数，从而确定选取的co_x, co_y DONE
                # 2. 计算邻近两个坐标，按dep的个位数计算比例/向量 分别增加x和y的数值 DONE

                # DONE 重新标定 40cm - 70cm
                # == mask大小随深度改变而改变==
                if init_Mi_coor==0:
                    if smooth_vertical == 1:
                        temp_x, temp_y = Mi_coor_smooth_vertical(Ca_coor, depth_base)
                    elif smooth_parallel == 1:
                        temp_x, temp_y = Mi_coor_smooth_parallel(Ca_coor, Im_x, Im_y, depth_base)
                    elif direct_mapping == 1:
                        co_x, co_y = fun_co(0.4, 'direct_mapping')
                        temp_x, temp_y = Mi_coor2(Im_x, Im_y, co_x, co_y)
                    else:
                        co_x, co_y = fun_co(0.4, 'only_one')  # 固定距离，固定位置（中心）
                        temp_x, temp_y = Mi_coor(Ca_coor, co_x, co_y)
                    Mi_x, Mi_y = temp_x, temp_y
                    # 偶然的深度值大幅波动会造成mask的位置跳动，因此使用以下if语句消除跳动
                    # if np.abs(temp_y[96]-Mi_y[96]) <= 200:
                    #     Mi_x, Mi_y = temp_x, temp_y
                    # if np.abs(temp_y[96]-Mi_y[96]) <= 200:
                    #     print(temp_y[68]-temp_y[59])
                else:
                    Mi_x = np.zeros(98).astype(int)
                    Mi_y = np.zeros(98).astype(int)
                    if direct_mapping == 1:
                        co_x, co_y = fun_co(0.4, 'direct_mapping')
                        Mi_x, Mi_y = Mi_coor2(Im_x, Im_y, co_x, co_y)
                    else:
                        co_x, co_y = fun_co(0.4, 'only_one')
                        Mi_x, Mi_y = Mi_coor(Ca_coor, co_x, co_y)
                    init_Mi_coor = 0
                # ==/ mask大小随深度改变而改变==

                
                # ==缩脸 - Mi扩==
                if contour_thin == 1:
                    co_fat = 3
                    for i in range(16):
                        # if Mi_x[i]<=Mp_height-co_fat and Mi_x[i]>=co_fat and Mi_y[i]<Mp_width-co_fat:
                        if Mi_x[i]>=co_fat and Mi_y[i]<=Mp_width-co_fat and Mi_x[i]<=Mp_height and Mi_y[i]>=0:
                            Mi_x[i] -= co_fat
                            Mi_y[i] += co_fat
                    if Mi_y[16]<Mp_width-co_fat and Mi_y[16]>=0:
                        Mi_y[16] += co_fat
                    for i in range(16, 33):
                        if Mi_x[i]<=Mp_height-co_fat and Mi_y[i]<=Mp_width-co_fat and Mi_x[i]>=0 and Mi_y[i]>=0:
                            Mi_x[i] += co_fat
                            Mi_y[i] += co_fat
                # ==/缩脸 - Mi扩==

                # DONE 曲线优化
                # 221010
                # 1. 提高分辨率 DONE
                # 2. 更换脸部显示图 DONE

                # ==曲线轮廓bezier curve==
                Mi_points_base = np.array([[Mi_x[i], Mi_y[i]] for i in range(len(Mi_x))])

                if schema_mask == 1:
                    # 图像像素坐标系
                    index_con_points = [4,8,12,16,20,24,28,46,45,50,51,38,34,33,4]
                    index_f1_points = [4,60,66,64,55,51,59,68,74,72,28]
                    index_f2_points = [8,66,55,57,59,74,24]
                    index_f3_points = [8,76,79,82,24]
                    index_f4_points = [12,76,85,82,20]

                    con_points = Im_points_base[index_con_points]
                    f1_points = Im_points_base[index_f1_points]
                    f2_points = Im_points_base[index_f2_points]
                    f3_points = Im_points_base[index_f3_points]
                    f4_points = Im_points_base[index_f4_points]
                    f5_points = Im_points_base[[33,60]]
                    f6_points = Im_points_base[[46,72]]
                    f7_points = Im_points_base[[12,85]]
                    f8_points = Im_points_base[[20,85]]
                    f9_points = Im_points_base[[16,85]]
                    f10_points = Im_points_base[[8,55]]
                    f11_points = Im_points_base[[59,24]]

                    curve_dif = 10
                    curve_con_points = evaluate_bezier(con_points, curve_dif).astype(int)
                    # curve_f1_points = evaluate_bezier(f1_points, curve_dif).astype(int)
                    # curve_f2_points = evaluate_bezier(f2_points, curve_dif).astype(int)
                    # curve_f3_points = evaluate_bezier(f3_points, curve_dif).astype(int)
                    # curve_f4_points = evaluate_bezier(f4_points, curve_dif).astype(int)

                    point_list_Im = [curve_con_points,
                        f1_points, f2_points, f3_points, f4_points,
                        # curve_f1_points, curve_f2_points, curve_f3_points, curve_f4_points,
                        f5_points, f6_points, f7_points, f8_points, f9_points, f10_points, f11_points]

                    # 镜面坐标系
                    con_points = Mi_points_base[index_con_points]
                    f1_points = Mi_points_base[index_f1_points]
                    f2_points = Mi_points_base[index_f2_points]
                    f3_points = Mi_points_base[index_f3_points]
                    f4_points = Mi_points_base[index_f4_points]
                    f5_points = Mi_points_base[[33,60]]
                    f6_points = Mi_points_base[[46,72]]
                    f7_points = Mi_points_base[[12,85]]
                    f8_points = Mi_points_base[[20,85]]
                    f9_points = Mi_points_base[[16,85]]
                    f10_points = Mi_points_base[[8,55]]
                    f11_points = Mi_points_base[[59,24]]

                    curve_dif = 10
                    curve_con_points = evaluate_bezier(con_points, curve_dif).astype(int)
                    # curve_f1_points = evaluate_bezier(f1_points, curve_dif).astype(int)
                    # curve_f2_points = evaluate_bezier(f2_points, curve_dif).astype(int)
                    # curve_f3_points = evaluate_bezier(f3_points, curve_dif).astype(int)
                    # curve_f4_points = evaluate_bezier(f4_points, curve_dif).astype(int)

                    point_list_Mi = [curve_con_points,
                        f1_points, f2_points, f3_points, f4_points,
                        # curve_f1_points, curve_f2_points, curve_f3_points, curve_f4_points,
                        f5_points, f6_points, f7_points, f8_points, f9_points, f10_points, f11_points]
                
                elif schema_mask == 1.1:
                    # 图像像素坐标系
                    # con_points = Im_points_base[[51, 37,35,33,0,4,8,12,16,20,24,28,32,46,44,42, 51]]  # [51, 38,34,33,0,4,8,12,16,20,24,28,32,46,45,50, 51]
                    con_points_1 = Im_points_base[[32,46,44,42, 51, 37,35,33,0]]
                    con_points_2 = Im_points_base[[0,4,8,12,16,20,24,28,32]]
                    points_eye_right = np.append(Im_points_base[60: 68], Im_points_base[60]).reshape(-1, 2)
                    points_eye_left = np.append(Im_points_base[68: 76], Im_points_base[68]).reshape(-1, 2)
                    points_mouth_outer = np.append(Im_points_base[76: 88], Im_points_base[76]).reshape(-1, 2)
                    points_mouth_inner = np.append(Im_points_base[88: 96], Im_points_base[88]).reshape(-1, 2)
                    Im_p01 = Im_points_base[[8, 66, 55, 51, 59, 74, 24]]
                    Im_p02 = Im_points_base[[8, 55, 57, 59, 24]]
                    Im_p03 = Im_points_base[[8, 76, 12, 85, 20, 82, 24]]
                    Im_p04_1 = Im_points_base[[0, 60, 4]]
                    Im_p04_2 = Im_points_base[[32, 72, 28]]
                    Im_p05_1 = Im_points_base[[55, 64]]
                    Im_p05_2 = Im_points_base[[59, 68]]
                    Im_p06 = Im_points_base[[16, 85]]
                    Im_p07_1 = Im_points_base[[33, 62, 37]]
                    Im_p07_2 = Im_points_base[[46, 70, 42]]

                    curve_dif = 10
                    # curve_con_points = evaluate_bezier(con_points, curve_dif).astype(int)
                    curve_con_points_1 = evaluate_bezier(con_points_1, curve_dif).astype(int)
                    curve_con_points_2 = evaluate_bezier(con_points_2, curve_dif).astype(int)
                    curve_eye_right = evaluate_bezier(points_eye_right, curve_dif).astype(int)
                    curve_eye_left = evaluate_bezier(points_eye_left, curve_dif).astype(int)
                    curve_mouth_outer = evaluate_bezier(points_mouth_outer, curve_dif).astype(int)
                    curve_mouth_inner = evaluate_bezier(points_mouth_inner, curve_dif).astype(int)
                    # curve_p01 = evaluate_bezier(Im_p01).astype(int)
                    # curve_p02 = evaluate_bezier(Im_p02).astype(int)
                    # curve_p03 = evaluate_bezier(Im_p03).astype(int)

                    point_list_Im = [
                        # curve_con_points,
                        curve_con_points_1,
                        curve_con_points_2,
                        # curve_p01, curve_p01, curve_p01,
                        Im_p01, Im_p02, Im_p03,
                        Im_p04_1, Im_p04_2, Im_p05_1, Im_p05_2, Im_p06,
                        Im_p07_1, Im_p07_2,
                        curve_eye_right, curve_eye_left, curve_mouth_outer, curve_mouth_inner]

                    # 镜面坐标系
                    # con_points = Mi_points_base[[51, 37,35,33,0,4,8,12,16,20,24,28,32,46,44,42, 51]]  # [51, 38,34,33,0,4,8,12,16,20,24,28,32,46,45,50, 51]
                    con_points_1 = Mi_points_base[[32,46,44,42, 51, 37,35,33,0]]
                    con_points_2 = Mi_points_base[[0,4,8,12,16,20,24,28,32]]
                    points_eye_right = np.append(Mi_points_base[60: 68], Mi_points_base[60]).reshape(-1, 2)
                    points_eye_left = np.append(Mi_points_base[68: 76], Mi_points_base[68]).reshape(-1, 2)
                    points_mouth_outer = np.append(Mi_points_base[76: 88], Mi_points_base[76]).reshape(-1, 2)
                    points_mouth_inner = np.append(Mi_points_base[88: 96], Mi_points_base[88]).reshape(-1, 2)
                    Mi_p01 = Mi_points_base[[8, 66, 55, 51, 59, 74, 24]]
                    Mi_p02 = Mi_points_base[[8, 55, 57, 59, 24]]
                    Mi_p03 = Mi_points_base[[8, 76, 12, 85, 20, 82, 24]]
                    Mi_p04_1 = Mi_points_base[[0, 60, 4]]
                    Mi_p04_2 = Mi_points_base[[32, 72, 28]]
                    Mi_p05_1 = Mi_points_base[[55, 64]]
                    Mi_p05_2 = Mi_points_base[[59, 68]]
                    Mi_p06 = Mi_points_base[[16, 85]]
                    Mi_p07_1 = Mi_points_base[[33, 62, 37]]
                    Mi_p07_2 = Mi_points_base[[46, 70, 42]]

                    curve_dif = 10
                    # curve_con_points = evaluate_bezier(con_points, curve_dif).astype(int)
                    curve_con_points_1 = evaluate_bezier(con_points_1, curve_dif).astype(int)
                    curve_con_points_2 = evaluate_bezier(con_points_2, curve_dif).astype(int)
                    curve_eye_right = evaluate_bezier(points_eye_right, curve_dif).astype(int)
                    curve_eye_left = evaluate_bezier(points_eye_left, curve_dif).astype(int)
                    curve_mouth_outer = evaluate_bezier(points_mouth_outer, curve_dif).astype(int)
                    curve_mouth_inner = evaluate_bezier(points_mouth_inner, curve_dif).astype(int)
                    # curve_p01 = evaluate_bezier(Im_p01).astype(int)
                    # curve_p02 = evaluate_bezier(Im_p02).astype(int)
                    # curve_p03 = evaluate_bezier(Im_p03).astype(int)

                    point_list_Mi = [
                        # curve_con_points,
                        curve_con_points_1,
                        curve_con_points_2,
                        # curve_p01, curve_p01, curve_p01,
                        Mi_p01, Mi_p02, Mi_p03,
                        Mi_p04_1, Mi_p04_2, Mi_p05_1, Mi_p05_2, Mi_p06,
                        Mi_p07_1, Mi_p07_2,
                        curve_eye_right, curve_eye_left, curve_mouth_outer, curve_mouth_inner]
                
                elif schema_mask == 2:
                    # 图像像素坐标系
                    points_contour = Im_points_base[0:33]
                    points_eyebrow_right = np.append(Im_points_base[33: 42], Im_points_base[33]).reshape(-1, 2)
                    points_eyebrow_left = np.append(Im_points_base[42: 51], Im_points_base[42]).reshape(-1, 2)
                    points_eye_right = np.append(Im_points_base[60: 68], Im_points_base[60]).reshape(-1, 2)
                    points_eye_left = np.append(Im_points_base[68: 76], Im_points_base[68]).reshape(-1, 2)
                    points_mouth_outer = np.append(Im_points_base[76: 88], Im_points_base[76]).reshape(-1, 2)
                    points_mouth_inner = np.append(Im_points_base[88: 96], Im_points_base[88]).reshape(-1, 2)
                    points_nose = np.append(np.append(Im_points_base[51], Im_points_base[55:60]), Im_points_base[51]).reshape(-1, 2)

                    curve_dif = 10
                    curve_contour = evaluate_bezier(points_contour, curve_dif)
                    curve_eyebrow_right = evaluate_bezier(points_eyebrow_right, curve_dif)
                    curve_eyebrow_left = evaluate_bezier(points_eyebrow_left, curve_dif)
                    curve_eye_right = evaluate_bezier(points_eye_right, curve_dif)
                    curve_eye_left = evaluate_bezier(points_eye_left, curve_dif)
                    curve_mouth_outer = evaluate_bezier(points_mouth_outer, curve_dif)
                    curve_mouth_inner = evaluate_bezier(points_mouth_inner, curve_dif)
                    curve_nose = evaluate_bezier(points_nose, curve_dif)

                    curve_contour = curve_contour.astype(int)
                    curve_eyebrow_right = curve_eyebrow_right.astype(int)
                    curve_eyebrow_left = curve_eyebrow_left.astype(int)
                    curve_eye_right = curve_eye_right.astype(int)
                    curve_eye_left = curve_eye_left.astype(int)
                    curve_mouth_outer = curve_mouth_outer.astype(int)
                    curve_mouth_inner = curve_mouth_inner.astype(int)
                    curve_nose = curve_nose.astype(int)
                    # curve_contour = np.unique(curve_contour, axis=0)
                    # curve_eyebrow_right = np.unique(curve_eyebrow_right, axis=0)
                    # curve_eyebrow_left = np.unique(curve_eyebrow_left, axis=0)
                    # curve_eye_right = np.unique(curve_eye_right, axis=0)
                    # curve_eye_left = np.unique(curve_eye_left, axis=0)
                    # curve_mouth_outer = np.unique(curve_mouth_outer, axis=0)
                    # curve_mouth_inner = np.unique(curve_mouth_inner, axis=0)
                    # curve_nose = np.unique(curve_nose, axis=0)
                    point_list_Im = [curve_contour, curve_eyebrow_right, curve_eyebrow_left,
                                    curve_eye_right, curve_eye_left,
                                    curve_mouth_outer, curve_mouth_inner,
                                    curve_nose]

                    # 镜面坐标系
                    Mpb = Mi_points_base
                    # Mi_points_contour = Mi_points_base[0:33]
                    Mi_points_contour = np.array([Mpb[0], Mpb[4], Mpb[8], Mpb[12], Mpb[16], Mpb[20], Mpb[24], Mpb[28], Mpb[32]])
                    # Mi_points_contour = np.array([Mpb[0], Mpb[4], Mpb[8], Mpb[16], Mpb[24], Mpb[28], Mpb[32]])
                    Mi_points_eyebrow_right = np.append(Mi_points_base[33: 42], Mi_points_base[33]).reshape(-1, 2)
                    Mi_points_eyebrow_left = np.append(Mi_points_base[42: 51], Mi_points_base[42]).reshape(-1, 2)
                    Mi_points_eye_right = np.append(Mi_points_base[60: 68], Mi_points_base[60]).reshape(-1, 2)
                    Mi_points_eye_left = np.append(Mi_points_base[68: 76], Mi_points_base[68]).reshape(-1, 2)
                    Mi_points_mouth_outer = np.append(Mi_points_base[76: 88], Mi_points_base[76]).reshape(-1, 2)
                    Mi_points_mouth_inner = np.append(Mi_points_base[88: 96], Mi_points_base[88]).reshape(-1, 2)
                    Mi_points_nose = np.append(np.append(Mi_points_base[51], Mi_points_base[55:60]), Mi_points_base[51]).reshape(-1, 2)

                    curve_dif = 10
                    Mi_curve_contour = evaluate_bezier(Mi_points_contour, curve_dif)
                    Mi_curve_eyebrow_right = evaluate_bezier(Mi_points_eyebrow_right, curve_dif)
                    Mi_curve_eyebrow_left = evaluate_bezier(Mi_points_eyebrow_left, curve_dif)
                    Mi_curve_eye_right = evaluate_bezier(Mi_points_eye_right, curve_dif)
                    Mi_curve_eye_left = evaluate_bezier(Mi_points_eye_left, curve_dif)
                    Mi_curve_mouth_outer = evaluate_bezier(Mi_points_mouth_outer, curve_dif)
                    Mi_curve_mouth_inner = evaluate_bezier(Mi_points_mouth_inner, curve_dif)
                    Mi_curve_nose = evaluate_bezier(Mi_points_nose, curve_dif)

                    Mi_curve_contour = Mi_curve_contour.astype(int)
                    Mi_curve_eyebrow_right = Mi_curve_eyebrow_right.astype(int)
                    Mi_curve_eyebrow_left = Mi_curve_eyebrow_left.astype(int)
                    Mi_curve_eye_right = Mi_curve_eye_right.astype(int)
                    Mi_curve_eye_left = Mi_curve_eye_left.astype(int)
                    Mi_curve_mouth_outer = Mi_curve_mouth_outer.astype(int)
                    Mi_curve_mouth_inner = Mi_curve_mouth_inner.astype(int)
                    Mi_curve_nose = Mi_curve_nose.astype(int)
                    # Mi_curve_contour = np.unique(Mi_curve_contour, axis=0)
                    # Mi_curve_eyebrow_right = np.unique(Mi_curve_eyebrow_right, axis=0)
                    # Mi_curve_eyebrow_left = np.unique(Mi_curve_eyebrow_left, axis=0)
                    # Mi_curve_eye_right = np.unique(Mi_curve_eye_right, axis=0)
                    # Mi_curve_eye_left = np.unique(Mi_curve_eye_left, axis=0)
                    # Mi_curve_mouth_outer = np.unique(Mi_curve_mouth_outer, axis=0)
                    # Mi_curve_mouth_inner = np.unique(Mi_curve_mouth_inner, axis=0)
                    # Mi_curve_nose = np.unique(Mi_curve_nose, axis=0)
                    point_list_Mi = [Mi_curve_contour, Mi_curve_eyebrow_right, Mi_curve_eyebrow_left,
                                    Mi_curve_eye_right, Mi_curve_eye_left,
                                    Mi_curve_mouth_outer, Mi_curve_mouth_inner,
                                    Mi_curve_nose]

                # ==镜面坐标系==

                if schema_mask == 1:
                    pass
                    # index_con_points = [4,8,12,16,20,24,28,46,45,50,51,38,34,33,4]
                    # index_f1_points = [4,60,66,64,55,51,59,68,74,72,28]
                    # index_f2_points = [8,66,55,57,59,74,24]
                    # index_f3_points = [8,76,79,82,24]
                    # index_f4_points = [12,76,85,82,20]
                elif schema_mask == 2:
                    pass
                # ==镜面坐标系==
                # ==/曲线轮廓bezier curve==

                # DONE cv2画弧线
                bezier_curve = 1
                if bezier_curve == 1:
                    # con_points = [4,8,12,16,20,24,28,46,45,50,51,38,34,33,4]
                    # f1_points = [4,60,66,64,55,51,59,68,74,72,28]
                    # f2_points = [8,66,55,57,59,74,24]
                    # f3_points = [8,76,79,82,24]
                    # f4_points = [12,76,85,82,20]
                    # f5_points = []

                    # list_len = len(line_list)
                    # p_list = 
                    # print(curve_contour)
                    for points in point_list_Im:
                        c_len = len(points)
                        for i in range(c_len-1):
                            # 处理图像窗口
                            Im_xc, Im_yc = points[i][0], points[i][1]
                            Im_xn, Im_yn = points[i+1][0], points[i+1][1]
                            cv2.line(frame, (Im_xc, Im_yc), (Im_xn, Im_yn), (137,190,178), 1)
                            # try:
                            #     cv2.line(frame, (Im_xc, Im_yc), (Im_xn, Im_yn), (137,190,178), 1)
                            # except error as err:
                            #     print(points[i], points[i])
                            #     print(points[i+1], points[i+1])
                            #     print(err)
                            # cv2.circle(frame, (Im_xc, Im_yc), 1, (0, 0, 255), 2)

                            # 处理黑屏窗口，即镜面窗口
                            # Mi_xc, Mi_yc = Mi_x[l[i]], Mi_y[l[i]]
                            # Mi_xn, Mi_yn = Mi_x[l[i+1]], Mi_y[l[i+1]]
                            # cv2.line(Mi_p, (Mi_xc, Mi_yc), (Mi_xn, Mi_yn), (137,190,178), 4)
                            # cv2.circle(Mi_p, (Mi_xc, Mi_yc), 1, (0, 0, 255), 2)
                    for points in point_list_Mi:
                        c_len = len(points)
                        for i in range(c_len-1):
                            # 处理黑屏窗口，即镜面窗口
                            Mi_xc, Mi_yc = points[i][0], points[i][1]
                            Mi_xn, Mi_yn = points[i+1][0], points[i+1][1]
                            cv2.line(Mi_p, (Mi_xc, Mi_yc), (Mi_xn, Mi_yn), (137,190,178), 4)

                # text = "%.2f depth"%depth_base
                # print(depth_base)
                # text = "{:.4f} depth".format(depth_base)
                # frame =  cv2.putText(frame, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # Mi_p =  cv2.putText(Mi_p, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                mask = 0
                if mask == 1:
                    con_points = [4,8,12,16,20,24,28,46,45,50,51,38,34,33,4]
                    f1_points = [4,60,66,64,55,51,59,68,74,72,28]
                    f2_points = [8,66,55,57,59,74,24]
                    f3_points = [8,76,79,82,24]
                    f4_points = [12,76,85,82,20]
                    # f5_points = []
                    line_list = [con_points,f1_points,f2_points,f3_points,f4_points,[33,60],[46,72],[12,85],[20,85],[16,85],[8,55],[59,24]]
                    # list_len = len(line_list)
                    # p_list = 
                    for l in line_list:
                        c_len = len(l)
                        for i in range(c_len-1):
                            # 处理图像窗口
                            Im_xc, Im_yc = Im_x[l[i]], Im_y[l[i]]
                            Im_xn, Im_yn = Im_x[l[i+1]], Im_y[l[i+1]]
                            cv2.line(frame, (Im_xc, Im_yc), (Im_xn, Im_yn), (137,190,178), 1)
                            cv2.circle(frame, (Im_xc, Im_yc), 1, (0, 0, 255), 2)

                            # 处理黑屏窗口，即镜面窗口
                            Mi_xc, Mi_yc = Mi_x[l[i]], Mi_y[l[i]]
                            Mi_xn, Mi_yn = Mi_x[l[i+1]], Mi_y[l[i+1]]
                            cv2.line(Mi_p, (Mi_xc, Mi_yc), (Mi_xn, Mi_yn), (137,190,178), 4)
                            cv2.circle(Mi_p, (Mi_xc, Mi_yc), 1, (0, 0, 255), 2)
                    # for l in line_list:
                    #     c_len = len(l)
                    #     for i in range(c_len-1):
                    #         x_pred_c = x_pred[l[i]]#lms_pred_merge[l[i%c_len]*2] * det_width
                    #         y_pred_c = y_pred[l[i]]#lms_pred_merge[l[i%c_len]*2+1] * det_height

                    #         x_pred_n = x_pred[l[i+1]]#lms_pred_merge[l[(i+1)%c_len]*2] * det_width
                    #         y_pred_n = y_pred[l[i+1]]#lms_pred_merge[l[(i+1)%c_len]*2+1] * det_height

                    #         cv2.line(frame, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), (int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin), (137,190,178), 1)
                    #         cv2.circle(frame, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), 1, (0, 0, 255), 2)

                    #         # cv2.line(Mi_p, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), (int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin), (137,190,178), 4)
                    #         # cv2.circle(Mi_p, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), 1, (0, 0, 255), 2)
                    #         ###current
                    #         Im_xc, Im_yc = int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin
                    #         Im_xn, Im_yn = int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin

                    #         dep, ca_coor = get_3d_camera_coordinate([Im_xc, Im_yc], aligned_depth_frame,depth_intrin)
                    #         [Ca_xc, Ca_yc, Ca_zc] =[*ca_coor]
                    #         Mi_xc = px(Ca_xc)
                    #         Mi_yc = np.dot(np.array([1,Ca_yc,Ca_zc]),co_y)

                    #         ###next
                    #         dep, ca_coor = get_3d_camera_coordinate([Im_xn, Im_yn], aligned_depth_frame,depth_intrin)
                    #         [Ca_xn, Ca_yn, Ca_zn] =[*ca_coor]
                    #         Mi_xn = px(Ca_xn)
                    #         Mi_yn = np.dot(np.array([1,Ca_yn,Ca_zn]),co_y)

                    #         cv2.line(Mi_p, (int(Mi_xc), int(Mi_yc)), (int(Mi_xn), int(Mi_yn)), (137,190,178), 4)
                    #         cv2.circle(Mi_p, (int(Mi_xc), int(Mi_yc)), 1, (0, 0, 255), 2)





                    

                # else:   
                #     for i in range(cfg.num_lms):
                #         x_pred = lms_pred_merge[i*2] * det_width
                #         y_pred = lms_pred_merge[i*2+1] * det_height
                #         # text = "(" + str(i) + ")"
                #         cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
                #         cv2.circle(Mi_p, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)



                        # cv2.putText(frame, text, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

                        # mask = 1
                        # if mask == 1:
                        #     con_pionts = [4,8,12,16,20,24,28,46,45,50,38,34,33]
                        #     if ii
                        # else:
                        #     cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)

                        # cv2.putText(frame, text, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1)
                
            # count += 1

            # cv2.setMouseCallback('RealSense', on_EVENT_BUTTONDOWN)
            # cv2.setMouseCallback('Mirror_P', on_EVENT_BUTTONDOWN2)


            frame = cv2.flip(frame, 1)
            text = "{:.4f} depth".format(depth_text)
            frame =  cv2.putText(frame, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            Mi_p =  cv2.putText(Mi_p, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow('RealSense', frame)
            cv2.imshow('Mirror_P', Mi_p)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                # print(re)
                # dataframe = pd.DataFrame({'Im_x': im_x, 'Im_y': im_y, 'Ca_x': ca_x, 'Ca_y': ca_y, 'Ca_z': ca_z, 'Mi_x': mi_x, 'Mi_y': mi_y} )
                # # print(dataframe)
                # dataframe.to_csv(r"D:\13219\Downloads\camera_calibration_tool-master\caliberate4\test.csv", index=False, sep=',')
                break
            
    finally:
        # Stop streaming
        pipeline.stop()


    # if video_file == 'camera':
    #     cap = cv2.VideoCapture(0)  # 参数0，打开内置摄像头
    # else:
    #     cap = cv2.VideoCapture(video_file)
    # if (cap.isOpened()== False): 
    #     print("Error opening video stream or file")
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('out2.avi',fourcc,20.0,(width,height))

    # custom_width = 480
    # custom_length = 640
    # # custom_width = 960
    # # custom_length = 1280

    # # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (1920,1200))
    # # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (600,600))
    # # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (480,640))
    # # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (custom_width, custom_length))
    # # count = 0
    # while(cap.isOpened()):
    #     flag, frame = cap.read()  # frame是图片 frame.shape = (480, 640, 3)
    #     # ###
    #     # cv2.namedWindow("frame")
    #     # frame = cv2.resize(frame,(1920, 1200))
    #     # frame_predict = np.zeros((1920, 1200, 3), np.uint8)
    #     # frame_predict = np.zeros((480, 640, 3), np.uint8)
    #     # frame = cv2.resize(frame,(600,600))
    #     # frame_predict = np.zeros((600, 600, 3), np.uint8)

    #     # try:
    #     #     frame = cv2.resize(frame,(custom_width, custom_length))
    #     # except:
    #     #     continue
    #     frame_predict = np.zeros((custom_length, custom_width, 3), np.uint8)  ########################## 这里长和宽是反的
    #     # frame_predict = np.zeros((custom_width, custom_length, 3), np.uint8)

    #     frame_predict.fill(0)  # 设置背景色
    #     # cv2.namedWindow("frame")
    #     cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #     cv2.namedWindow("frame_predict", cv2.WINDOW_NORMAL)  # WINDOW_AUTOSIZE
        # cv2.resizeWindow("frame",600,600)
        # cv2.resizeWindow("frame",600,600)
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(width,height)
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # # out = cv2.VideoWriter('out2.avi',fourcc,20.0,(width,height))

        # out = cv2.VideoWriter('out2.avi',fourcc,20.0,(600,600))
        # out = cv2.VideoWriter('out2.avi',fourcc,fps,(600,600))

        # ##
        # print("shape is", frame.shape)
        # print("0度是", frame.shape[0])
        # print("1度是", frame.shape[1])
        # break
        # if flag == True:
        #     detections, _ = detector.detect(frame, my_thresh, 1)
        #     for i in range(len(detections)):
        #         det_xmin = detections[i][2]
        #         det_ymin = detections[i][3]
        #         det_width = detections[i][4]
        #         det_height = detections[i][5]
        #         det_xmax = det_xmin + det_width - 1
        #         det_ymax = det_ymin + det_height - 1

        #         det_xmin -= int(det_width * (det_box_scale-1)/2)
        #         # remove a part of top area for alignment, see paper for details
        #         det_ymin += int(det_height * (det_box_scale-1)/2)
        #         det_xmax += int(det_width * (det_box_scale-1)/2)
        #         det_ymax += int(det_height * (det_box_scale-1)/2)
        #         det_xmin = max(det_xmin, 0)
        #         det_ymin = max(det_ymin, 0)
        #         det_xmax = min(det_xmax, frame_width-1)
        #         det_ymax = min(det_ymax, frame_height-1)
        #         det_width = det_xmax - det_xmin + 1
        #         det_height = det_ymax - det_ymin + 1
        #         # cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        #         det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
        #         det_crop = cv2.resize(det_crop, (input_size, input_size))
        #         inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        #         inputs = preprocess(inputs).unsqueeze(0)
        #         inputs = inputs.to(device)
        #         lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
        #         lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        #         tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        #         tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        #         tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        #         tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        #         lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        #         lms_pred = lms_pred.cpu().numpy()
        #         lms_pred_merge = lms_pred_merge.cpu().numpy()

        #         mask = 1

        #         if mask == 1:
        #             con_points = [4,8,12,16,20,24,28,46,45,50,51,38,34,33,4]
        #             f1_points = [4,60,66,64,55,51,59,68,74,72,28]
        #             f2_points = [8,66,55,57,59,74,24]
        #             f3_points = [8,76,79,82,24]
        #             f4_points = [12,76,85,82,20]
        #             # f5_points = []
        #             line_list = [con_points,f1_points,f2_points,f3_points,f4_points,[33,60],[46,72],[12,85],[20,85],[16,85],[8,55],[59,24]]
        #             list_len = len(line_list)
        #             # p_list = 
        #             for l in line_list:
        #                 c_len = len(l)
        #                 for i in range(c_len-1):
        #                     x_pred_c = lms_pred_merge[l[i%c_len]*2] * det_width
        #                     y_pred_c = lms_pred_merge[l[i%c_len]*2+1] * det_height

        #                     x_pred_n = lms_pred_merge[l[(i+1)%c_len]*2] * det_width
        #                     y_pred_n = lms_pred_merge[l[(i+1)%c_len]*2+1] * det_height

        #                     cv2.line(frame, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), (int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin), (137,190,178), 1)
        #                     cv2.circle(frame, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), 1, (0, 0, 255), 2)
        #                     cv2.line(frame_predict, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), (int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin), (137,190,178), 4)
        #                     cv2.circle(frame_predict, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), 1, (0, 0, 255), 2)


                    

        #         else:   
        #             for i in range(cfg.num_lms):
        #                 x_pred = lms_pred_merge[i*2] * det_width
        #                 y_pred = lms_pred_merge[i*2+1] * det_height
        #                 # text = "(" + str(i) + ")"
        #                 cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
        #                 cv2.circle(frame_predict, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
        #                 # cv2.putText(frame, text, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

        #                 # mask = 1
        #                 # if mask == 1:
        #                 #     con_pionts = [4,8,12,16,20,24,28,46,45,50,38,34,33]
        #                 #     if ii
        #                 # else:
        #                 #     cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)

        #                 # cv2.putText(frame, text, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1)
                
        #     count += 1

        #     # cv2.imwrite('video_out2/'+str(count)+'.jpg', frame)
        #     # cv2.namedWindow("frame")
        #     # frame = cv2.resize(frame,(600,600))
        #     # cv2.resizeWindow("frame",600,600)

        #     # out.write(frame)
        #     out.write(frame_predict)

        #     cv2.imshow('frame', frame)
        #     cv2.imshow('frame_predict', frame_predict)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # else:
        #     break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

demo_video(video_file, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
