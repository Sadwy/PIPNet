import numpy as np
import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
# import absolutu_Orientation_Quaternion as aoq


# file_path = r'C:\Users\221\Desktop\engineering\caliberate\batch data\batch data\100_right.csv'
# dir_path = r'D:\13219\Desktop\camera_calibration_tool-master\caliberate4\batch221203'
# file_path = r'D:\13219\Desktop\camera_calibration_tool-master\caliberate4\batch221202\30_left.csv'


def calcu_co(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)
    dim = data.shape[0]
    # im_x, im_y = data[:,0], data[:,1]
    ca_x, ca_y, ca_z = data[:,2], data[:,3], data[:,4]
    mi_x, mi_y = data[:,5],data[:,6]



    f1 = np.polyfit(ca_x, mi_x, 1)
    print("co_x:\n[{0}, {1}]".format(f1[0], f1[1]))
    # p1 = np.poly1d(f1)
    # print("p1:", p1)
    # y_vals = p1(ca_x)

    ####AX = B
    A = np.vstack((np.ones_like(ca_y),ca_y,ca_z))
    mi_y = np.resize(mi_y,(dim,1))
    # print(A.shape,mi_y.shape)
    result,_,_,_ = np.linalg.lstsq(A.T,mi_y)
    # print("co_y:",result.reshape(-1))
    co_y = result.reshape(-1)
    print("co_y:\n[{0}, {1}, {2}]".format(co_y[0], co_y[1], co_y[2]))
    # print(np.dot(A.T,result), mi_y)

    co_xy = np.append(f1, co_y).reshape(-1)

    # 保存结果到txt文件
    np.savetxt("./caliberate4/batch/{}.txt".format(file_path.split('\\')[-1].split('.')[0]),
                co_xy, delimiter=",", fmt='%.06f', newline=', ')

dir_path = r'D:\13219\Desktop\camera_calibration_tool-master\caliberate4'

if __name__ == '__main__':
    # for num in [40, 50, 60, 70]:
    #     print('\n', num, 'left')
    #     file_path = '{0}\\{1}_left.csv'.format(dir_path, num)
    #     calcu_co(file_path=file_path)

        # print('\n', num, 'right')
        # file_path = '{0}\\{1}_right.csv'.format(dir_path, num)
        # calcu_co(file_path=file_path)
    # for fname in ['central', 'upper_left', 'upper_right', 'lower_left', 'lower_right']:
    #     print('\n', fname, 'left')
    #     file_path = '{0}\\{1}.csv'.format(dir_path, fname)
    #     calcu_co(file_path=file_path)
    # co_40l_upper_left = np.loadtxt(r"D:\13219\Desktop\camera_calibration_tool-master\caliberate4\batch221209\upper_left.txt", delimiter=", ", dtype=float)
    # print(co_40l_upper_left)

    # 修改文件名，自动计算参数并生成txt文件
    file_path = '{0}\\name.csv'.format(dir_path)
    calcu_co(file_path=file_path)