import numpy as np
import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
# import absolutu_Orientation_Quaternion as aoq


# file_path = r'C:\Users\221\Desktop\engineering\caliberate\batch data\batch data\100_right.csv'
# file_path = r'D:\13219\Desktop\camera_calibration_tool-master\caliberate4\batch221202\30_left.csv'
file_path = r'D:\13219\Desktop\camera_calibration_tool-master\caliberate4\test.csv'


data = pd.read_csv(file_path)
#print(type(data))
data = np.array(data)
# print(data.shape)
dim = data.shape[0]
# print(dim)
im_x, im_y = data[:,0], data[:,1]
ca_x, ca_y, ca_z = data[:,2], data[:,3], data[:,4]
mi_x, mi_y = data[:,5],data[:,6]



f1 = np.polyfit(ca_x, mi_x, 1)
print("co_x:",f1)
p1 = np.poly1d(f1)
# print("p1:", p1)
y_vals = p1(ca_x)

####AX = B
##B ：mi_y  A: [1, y, z]
# print(mi_y.shape)
A = np.vstack((np.ones_like(ca_y),ca_y,ca_z))
# print(B.shape,B)
mi_y = np.resize(mi_y,(dim,1))
# print(A.shape,mi_y.shape)
result,_,_,_ = np.linalg.lstsq(A.T,mi_y)
print("co_y:",result.reshape(-1))
# print(np.dot(A.T,result), mi_y)


co_x = [-4323.70664676, 1041.6113063]
px = np.poly1d(co_x)

co_y = [218.9787, 4133.2564, 1197.3876]
co_y = np.array(co_y).T

# print(np.dot(A.T,co_y))

p1 = np.poly1d(f1)
# print("p1:", p1)
y_vals = p1(ca_x)
# print(y_vals)


# plot1 = plt.plot(ca_x, mi_x, 's',label='original values')
# plot2 = plt.plot(ca_x, y_vals, 'r',label='polyfit values')
# plt.xlabel('ca_x')
# plt.ylabel('mi_x')
# plt.legend(loc=4) #指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()





# print(mi_y)
# table = data.sheet_by_name('Sheet1')
# # get position in the camera coordination
# c_x = table.col_values((2))[1:]
# c_y = table.col_values((3))[1:]
# c_z = table.col_values((4))[1:]
# c_x = np.array(c_x)
# c_y = np.array(c_y)
# c_z = np.array(c_z)
# camera_cor = np.vstack((c_x,c_y,c_z))
# # get position in the word coordination
# w_x = table.col_values((5))[1:]
# w_y = table.col_values((5))[1:]
# # w_z = table.col_values((5))[1:]
# w_x = np.array(w_x)
# w_y = np.array(w_y)
# # w_z = np.array(w_z)
# # word_cor = np.vstack((w_x,w_y,w_z))
# world_cor = np.vstack((w_x,w_y))
# # set the initial doScale = 1
# doScale = 1
# print(camera_cor, world_cor)