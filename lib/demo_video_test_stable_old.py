from tkinter.messagebox import NO, YES
print('may I find you, cv2?@demo_video.py')
import cv2, os
print('yes, you can.@demo_video.py')
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
sys.path.insert(0, r'..\FaceBoxesV2')
sys.path.insert(0, r'...')
# sys.path.insert(0, r'D:\13219\Desktop\PIPNet\FaceBoxesV2')
# sys.path.insert(0, r'D:\13219\Desktop\PIPNet')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

import pyrealsense2 as rs
import cv2
import csv
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *

# python lib/demo_video.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera
if not len(sys.argv) == 3:
    print('Format:')
    print('python lib/demo_video.py config_file video_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]  # pip_32_16_60_r18_l2_l1_10_1_nb10
data_name = sys.argv[1].split('/')[-2]  # WFLW
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)  # .experiments.WFLW.pip_32_16_60_r18_l2_l1_10_1_nb10
video_file = sys.argv[2]  # camera

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

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
co_x = [-4624.19868258, 986.23460526]  # 距离50cm
px = np.poly1d(co_x)
## 
#### mi_y = np.dot([1,ca_y,ca_z],co_y)
# co_y = [218.9787, 4133.2564, 1197.3876]  # abandon
# co_y = [-206.83741549, 4117.60761919, 1080.1952626]
co_y = [-47.91786684, 4183.69022911, 1011.09430934]  # 距离50cm
co_y = np.array(co_y).T
##




def demo_video(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.9
    det_box_scale = 1.2
    init_Ca_coor = 1

    net.eval()

    #### start realsense 2022.9.6
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流
    
    # config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    # config.enable_stream(rs.stream.depth,  1280, 720, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    pipe_profile = pipeline.start(config)  # streaming流开始
    
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
            #### 获取相机参数 ####
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

            if not aligned_depth_frame or not aligned_color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
 
            color_image = np.asanyarray(aligned_color_frame.get_data())
 
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            frame = color_image
            # Show images
            



            def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin, depth_base=0):
                x = depth_pixel[0]
                y = depth_pixel[1]
                dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
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



            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            
            Mp_length = 1200
            Mp_width = 1750
            Mi_p = np.zeros((Mp_length, Mp_width,3), np.uint8) 
            cv2.namedWindow('Mirror_P', cv2.WINDOW_AUTOSIZE)
            frame_width = 720
            frame_height = 1280

            ### detect
            detections, _ = detector.detect(frame, my_thresh, 1)
            for i in range(len(detections)):
                det_xmin = detections[i][2]
                det_ymin = detections[i][3]
                det_width = detections[i][4]
                det_height = detections[i][5]
                det_xmax = det_xmin + det_width - 1
                det_ymax = det_ymin + det_height - 1

                det_xmin -= int(det_width * (det_box_scale-1)/2)
                # remove a part of top area for alignment, see paper for details
                det_ymin += int(det_height * (det_box_scale-1)/2)
                det_xmax += int(det_width * (det_box_scale-1)/2)
                det_ymax += int(det_height * (det_box_scale-1)/2)
                det_xmin = max(det_xmin, 0)
                det_ymin = max(det_ymin, 0)
                det_xmax = min(det_xmax, frame_width-1)
                det_ymax = min(det_ymax, frame_height-1)
                det_width = det_xmax - det_xmin + 1
                det_height = det_ymax - det_ymin + 1
                # cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
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

                depth_base, _= get_3d_camera_coordinate([Im_x[97], Im_y[97]], aligned_depth_frame, depth_intrin)

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

                Ca_x, Ca_y, Ca_z = Ca_coor[:, 0], Ca_coor[:, 1], Ca_coor[:, 2]
                Mi_x = px(Ca_x).astype(int)
                Mi_y = np.dot(
                    np.array([np.array([1]*98), Ca_y, Ca_z]).T,
                    co_y
                    ).astype(int)

                mask = 1

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





                    

                else:   
                    for i in range(cfg.num_lms):
                        x_pred = lms_pred_merge[i*2] * det_width
                        y_pred = lms_pred_merge[i*2+1] * det_height
                        # text = "(" + str(i) + ")"
                        cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
                        cv2.circle(Mi_p, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
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
