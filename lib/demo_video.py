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

from networks import *
import data_utils
from functions import *

if not len(sys.argv) == 3:
    print('Format:')
    print('python lib/demo_video.py config_file video_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
video_file = sys.argv[2]

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
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('out2.avi',fourcc,20.0,(width,height))

    out = cv2.VideoWriter('out2.avi',fourcc,20.0,(600,600))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        # ###
        # cv2.namedWindow("frame")
        frame = cv2.resize(frame,(600,600))
        cv2.namedWindow("frame")
        cv2.resizeWindow("frame",600,600)
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
        if ret == True:
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


                    

                else:   
                    for i in range(cfg.num_lms):
                        x_pred = lms_pred_merge[i*2] * det_width
                        y_pred = lms_pred_merge[i*2+1] * det_height
                        # text = "(" + str(i) + ")"
                        cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
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

            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

demo_video(video_file, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
