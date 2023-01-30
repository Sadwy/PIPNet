import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
 

#  def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("1:",img[y][x][0])
#         print("2:",img[y][x][1])
#         print("3:",img[y][x][2])
 
#         xy = "%d,%d" % (x, y)
 
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)

# num_cali = 0

if __name__ == "__main__":
    # Configure depth and color streams
    # re = []
    im_x = []
    im_y = []
    ca_x = []
    ca_y = []
    ca_z = []
    mi_x = []
    mi_y = []
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流
    
    # config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    pipe_profile = pipeline.start(config)  # streaming流开始
    
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐

    num_cali = 0

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
            images = color_image
            # Show images
            




            def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
                x = depth_pixel[0]
                y = depth_pixel[1]
                dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
                # print ('depth: ',dis)       # 深度单位是m
                camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
                # print ('camera_coordinate: ',camera_coordinate)
                return dis, camera_coordinate

            def on_EVENT_BUTTONDOWN(event, x, y,flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # print("1:",images[y][x][0])
                    # print("2:",images[y][x][1])
                    # print("3:",images[y][x][2])
                    print("camera_click")

                    depth, ca_coor = get_3d_camera_coordinate([x,y],aligned_depth_frame,depth_intrin)
                    # print(depth, ca_coor)
            
                    # xy = "%d,%d,%f,%f,%f" % (x, y,*ca_coor)
                    # temp = [x,y,*ca_coor]
                    [a,b,c] = [*ca_coor]
                    # re.append(temp)
                    im_x.append(x)
                    im_y.append(y)
                    ca_x.append(a)
                    ca_y.append(b)
                    ca_z.append(c)

                    # result.append(xy)
            def on_EVENT_BUTTONDOWN2(event, x, y, flags, param):        
                global num_cali
                if event == cv2.EVENT_LBUTTONDOWN:
                    print("mirror_click")
                    mi_x.append(x)
                    mi_y.append(y)
                    num_cali += 1



            # cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            
            # Mp_length = 1200
            # Mp_width = 1750
            Mp_length = 1920#1920+10
            Mp_width = 1200#1200+20
            Mi_p = np.zeros((Mp_length, Mp_width,3), np.uint8)
            # cv2.namedWindow('Mirror_P', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Mirror_P',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Mirror_P', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.setMouseCallback('RealSense', on_EVENT_BUTTONDOWN)
            cv2.setMouseCallback('Mirror_P', on_EVENT_BUTTONDOWN2)


            cv2.imshow('RealSense', images)
            cv2.imshow('Mirror_P', Mi_p)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                # print(re)
                dataframe = pd.DataFrame({'Im_x': im_x, 'Im_y': im_y, 'Ca_x': ca_x, 'Ca_y': ca_y, 'Ca_z': ca_z, 'Mi_x': mi_x, 'Mi_y': mi_y} )
                # print(dataframe)
                dataframe.to_csv(r"D:\13219\Desktop\camera_calibration_tool-master\caliberate4\test.csv", index=False, sep=',')
                break

            # 此处控制每次采集的数量
            if 0==1 and num_cali==10:
                cv2.destroyAllWindows()
                # print(re)
                dataframe = pd.DataFrame({'Im_x': im_x, 'Im_y': im_y, 'Ca_x': ca_x, 'Ca_y': ca_y, 'Ca_z': ca_z, 'Mi_x': mi_x, 'Mi_y': mi_y} )
                # print(dataframe)
                dataframe.to_csv(r"D:\13219\Desktop\camera_calibration_tool-master\caliberate4\test.csv", index=False, sep=',')
                break
            
    finally:
        # Stop streaming
        pipeline.stop()
        