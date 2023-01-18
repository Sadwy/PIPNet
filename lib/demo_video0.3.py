import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pyrealsense2 as rs
import numpy as np

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

def demo_video(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.9
    det_box_scale = 1.2

    net.eval()
    if video_file == 'camera':
        # cap = cv2.VideoCapture(0)  # 参数0，打开内置摄像头
        frame_width = 640
        frame_height = 480
        pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
        config = rs.config()  # 定义配置config
        config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, 15)  # 配置depth流
        config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, 15)  # 配置color流
        
        pipe_profile = pipeline.start(config)  # streaming流开始
        
        # 创建对齐对象与color流对齐
        align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
        align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
    else:
        cap = cv2.VideoCapture(video_file)
    # if (cap.isOpened()== False): 
    #     print("Error opening video stream or file")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('out2.avi',fourcc,20.0,(width,height))

    custom_width = 600
    custom_length = 600
    # custom_width = 960
    # custom_length = 1280

    # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (1920,1200))
    # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (600,600))
    # out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (480,640))
    out = cv2.VideoWriter('out2.avi', fourcc, 20.0, (custom_width, custom_length))
    count = 0
    while True:
        flag, frame = cap.read()  # frame是图片 frame.shape = (480, 640, 3)
        # ###
        # cv2.namedWindow("frame")
        # frame = cv2.resize(frame,(1920, 1200))
        # frame_predict = np.zeros((1920, 1200, 3), np.uint8)
        # frame_predict = np.zeros((480, 640, 3), np.uint8)
        # frame = cv2.resize(frame,(600,600))
        # frame_predict = np.zeros((600, 600, 3), np.uint8)
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


        try:
            frame = cv2.resize(frame,(custom_width, custom_length))
        except:
            continue
        frame_predict = np.zeros((custom_length, custom_width, 3), np.uint8)  ########################## 这里长和宽是反的
        # frame_predict = np.zeros((custom_width, custom_length, 3), np.uint8)

        frame_predict.fill(0)  # 设置背景色
        # cv2.namedWindow("frame")
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame_predict", cv2.WINDOW_NORMAL)  # WINDOW_AUTOSIZE
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
        flag = 1
        if flag == True:
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

                mask = 1

                if mask == 1:
                    con_points = [4,8,12,16,20,24,28,46,45,50,51,38,34,33,4]
                    f1_points = [4,60,66,64,55,51,59,68,74,72,28]
                    f2_points = [8,66,55,57,59,74,24]
                    f3_points = [8,76,79,82,24]
                    f4_points = [12,76,85,82,20]
                    # f5_points = []
                    line_list = [con_points,f1_points,f2_points,f3_points,f4_points,[33,60],[46,72],[12,85],[20,85],[16,85],[8,55],[59,24]]
                    list_len = len(line_list)
                    # p_list = 
                    for l in line_list:
                        c_len = len(l)
                        for i in range(c_len-1):
                            x_pred_c = lms_pred_merge[l[i%c_len]*2] * det_width
                            y_pred_c = lms_pred_merge[l[i%c_len]*2+1] * det_height

                            x_pred_n = lms_pred_merge[l[(i+1)%c_len]*2] * det_width
                            y_pred_n = lms_pred_merge[l[(i+1)%c_len]*2+1] * det_height

                            cv2.line(frame, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), (int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin), (137,190,178), 1)
                            cv2.circle(frame, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), 1, (0, 0, 255), 2)
                            cv2.line(frame_predict, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), (int(x_pred_n)+det_xmin, int(y_pred_n)+det_ymin), (137,190,178), 4)
                            cv2.circle(frame_predict, (int(x_pred_c)+det_xmin, int(y_pred_c)+det_ymin), 1, (0, 0, 255), 2)


                    

                else:   
                    for i in range(cfg.num_lms):
                        x_pred = lms_pred_merge[i*2] * det_width
                        y_pred = lms_pred_merge[i*2+1] * det_height
                        # text = "(" + str(i) + ")"
                        cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
                        cv2.circle(frame_predict, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
                        # cv2.putText(frame, text, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

                        # mask = 1
                        # if mask == 1:
                        #     con_pionts = [4,8,12,16,20,24,28,46,45,50,38,34,33]
                        #     if ii
                        # else:
                        #     cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)

                        # cv2.putText(frame, text, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1)
                
            count += 1

            # cv2.imwrite('video_out2/'+str(count)+'.jpg', frame)
            # cv2.namedWindow("frame")
            # frame = cv2.resize(frame,(600,600))
            # cv2.resizeWindow("frame",600,600)

            # out.write(frame)
            out.write(frame_predict)

            cv2.imshow('frame', frame)
            cv2.imshow('frame_predict', frame_predict)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

demo_video(video_file, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
