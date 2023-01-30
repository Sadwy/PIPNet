import pyrealsense2 as rs
import numpy as np
import cv2
import csv
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



if __name__ == "__main__":
    # Configure depth and color streams
    re = []
    ca_x = []
    ca_y = []
    ca_z = [] 
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config

    custom_length = 1280
    custom_width = 720

    config.enable_stream(rs.stream.depth, custom_length, custom_width, rs.format.z16, 15)  # 配置depth流
    config.enable_stream(rs.stream.color, custom_length, custom_width, rs.format.bgr8, 15)  # 配置color流
    
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
            print("相机内参：", color_intrin)

            if not aligned_depth_frame or not aligned_color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            print(depth_image.shape)
 
            color_image = np.asanyarray(aligned_color_frame.get_data())
 
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))  # 
            # images = color_image
            # Show images
            




            def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
                x = depth_pixel[0]
                y = depth_pixel[1]
                dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
                # print ('depth: ',dis)       # 深度单位是m
                camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
                # print ('camera_coordinate: ',camera_coordinate)
                return dis, camera_coordinate

            def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print("1:",images[y][x][0])
                    print("2:",images[y][x][1])
                    print("3:",images[y][x][2])

                    depth, ca_coor = get_3d_camera_coordinate([x,y],aligned_depth_frame,depth_intrin)
                    print(depth, ca_coor)
            
                    xy = "%d,%d,%f,%f,%f" % (x, y,*ca_coor)
                    temp = [x,y,*ca_coor]
                    [a,b,c] = [*ca_coor]
                    re.append(temp)
                    ca_x.append(a)
                    ca_y.append(b)
                    ca_z.append(c)

            
                    cv2.circle(images, (x, y), 1, (255, 0, 0), thickness=-1)
                    cv2.putText(images, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0), thickness=1)
                    cv2.imshow("image", images)
                    # result.append(xy)


            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

            cv2.setMouseCallback('RealSense', on_EVENT_LBUTTONDOWN)



            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                print(re)
                dataframe = pd.DataFrame({'Ca_x': ca_x, 'Ca_y': ca_y, 'Ca_z': ca_z} )
                # print(dataframe)
                dataframe.to_csv(r"D:\13219\Downloads\camera_calibration_tool-master\caliberate3\test.csv", index=False, sep=',')
                break
            
    finally:
        # Stop streaming
        pipeline.stop()
        