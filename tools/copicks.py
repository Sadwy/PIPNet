import sys
import numpy as np

co_30l = np.loadtxt("../data/mapping/30l.txt", delimiter=",", dtype=float)
co_40l = np.loadtxt("../data/mapping/40l.txt", delimiter=",", dtype=float)
co_50l = np.loadtxt("../data/mapping/50l.txt", delimiter=",", dtype=float)
co_60l = np.loadtxt("../data/mapping/60l.txt", delimiter=",", dtype=float)

co_40l_central = np.loadtxt("../data/mapping/central.txt", delimiter=",", dtype=float)
co_40l_lower_left = np.loadtxt("../data/mapping/lower_left.txt", delimiter=",", dtype=float)
co_40l_lower_right = np.loadtxt("../data/mapping/lower_right.txt", delimiter=",", dtype=float)
co_40l_upper_left = np.loadtxt("../data/mapping/upper_left.txt", delimiter=",", dtype=float)
co_40l_upper_right = np.loadtxt("../data/mapping/upper_right.txt", delimiter=",", dtype=float)

co_only_one = np.loadtxt("../data/mapping/only_one.txt", delimiter=",", dtype=float)
co_direct_mapping = np.loadtxt("../data/mapping/direct_mapping.txt", delimiter=",", dtype=float)

# 在固定距离（40cm）标定了5个人脸，Im_anchor中存储的是5个人脸的左眼像素坐标
Im_anchor = np.array([
    [772,443],  # central
    [826,330],  # upper_left
    [664,352],  # upper_right
    [833,529],  # lower_left
    [661,518]  # lower_right
])

def co_x_y(case):
    '''
        存放标定参数
    '''
    if case == 'only_one':
        co_x, co_y = co_only_one[0:2], co_only_one[2:5]
    elif case == 'direct_mapping':
        co_x, co_y = co_direct_mapping[0:3], co_direct_mapping[3:6]
    elif case == 'all':  # abandon
        # eye_all
        co_x = [-4338.75931364, 1016.01427383]  # eye_all
        co_y = np.array([ -58.39654068, 4363.17551846, 480.81514565])  # eye_all
    elif case == '30l' or case == '30':
        # 30_left
        co_x = co_30l[0:2]#[-3619.8927068718267, 568.673747055739]#第一次20个数据#[-4411.216101963097, 513.5231620715755]
        co_y = co_30l[2:5]#[-94.81199768949638, 3962.6703460078015, 2194.3271830034096]#第一次20个数据#[287.2151328526499, 4762.105143813888, 119.16552468071754]
    elif case == '30r':
        # 30_right
        co_x = [-5162.9772815979495, 867.6911333376496]
        co_y = [464.3082865335194, 4659.06321356082, -403.8146710340726]
    elif case in ['40l_central', '40l_0']:
        co_x, co_y = co_40l_central[0:2], co_40l_central[2:5]
    elif case in ['40l_lower_left', '40l_3']:
        co_x, co_y = co_40l_lower_left[0:2], co_40l_lower_left[2:5]
    elif case in ['40l_lower_right', '40l_4']:
        co_x, co_y = co_40l_lower_right[0:2], co_40l_lower_right[2:5]
    elif case in ['40l_upper_left', '40l_1']:
        co_x, co_y = co_40l_upper_left[0:2], co_40l_upper_left[2:5]
    elif case in ['40l_upper_right', '40l_2']:
        co_x, co_y = co_40l_upper_right[0:2], co_40l_upper_right[2:5]
    elif case == '40l' or case == '40':
        # 40_left
        co_x = co_40l[0:2]#[-4235.192702805141, 563.0350350298273]#第二次20个数据#[-3534.4362540505113, 506.88231709362543]#第一次20个数据#[-3282.1276342033416, 525.8949665329701]
        co_y = co_40l[2:5]#[131.7932138049737, 4042.7152287199583, 1580.7375987676066]#第二次20个数据#[-209.68987183700813, 4160.539090455517, 2039.1860157250194]#第一次20个数据#[383.16043001418626, 4651.153325298742, 411.68414965410375]
        # co_x = [4279.63830441, 471.53709342]  # 朱老师标注 5 samples
        # co_y = [ -11.05739493, 4636.30685208, 1520.95591306]  # 朱老师标注 5 samples
    elif case == '40r':
        # 40_right
        co_x = [-4323.214171348305, 845.7245054032703]
        co_y = [513.0580302064942, 4325.008666614477, 417.18280431484703]
    elif case == '50l' or case == '50':
        # 50_left
        co_x = co_50l[0:2]#[-4277.972115098041, 559.5514804593623]#第二次20个数据#[-3940.238350579997, 596.7619885082566]#第一次20个数据#[-4558.692597875793, 551.5813767145238]
        co_y = co_50l[2:5]#[130.62288513513215, 4284.339166642898, 1553.1173509749397]#第二次20个数据#[230.37609477328593, 4304.42339518822, 1293.9679744215807]#第一次20个数据#[592.5253152137258, 4514.315083792963, 229.93276623290222]
    elif case == '50r':
        # 50_right
        co_x = [-4289.922867965609, 791.9183712404624]
        co_y = [930.9646978564291, 4501.852638696483, -74.32660026308258]
    elif case == '60l' or case == '60':
        # 60_left
        co_x = co_60l[0:2]#[-4070.88165315667, 594.2024305951276]#第二次20个数据#[-4169.415855908636, 545.2068896415399]#第一次20个数据#[-4464.9440408410355, 586.3793250846651]
        co_y = co_60l[2:5]#[104.1013357160276, 4218.699399176064, 1598.529448727265]#第二次20个数据#[-236.60446303114347, 3799.58293321013, 2119.9683799072805]#第一次20个数据#[-436.46599756624306, 4626.775363572502, 2298.3802612698732]
    elif case == '60r':
        # 60_right
        co_x = [-4092.2096224340225, 727.1869992794564]
        co_y = [502.0045861411944, 4541.568109790158, 843.925638091958]
    else:
        sys.exit("The case not stored!")
    return co_x, co_y

def fun_neighbour(Im_x, Im_y):  # abandon
    im = np.array([Im_x[79], Im_y[79]])  # 人中
    dis = np.sum(np.square(Im_anchor - im), axis=1)
    anchor_index = list(dis).index(np.min(dis))
    return anchor_index

def fun_co(dep, tag, left_or_right='l'):
    '''
        根据距离(dep)+在镜子中的位置(tag)+左右眼(默认左眼)，调取对应的标定参数
    '''
    # neigh = fun_neighbour(Im_x, Im_y)
    if tag == 'direct_mapping':
        co_x, co_y = co_x_y('direct_mapping')
    elif tag == 'only_one':
        co_x, co_y = co_x_y('only_one')
    else:
        dep = int(dep*10)
        if dep <= 3:
            dep = 3
        elif dep >= 6:
            dep = 6
        co_x, co_y = co_x_y('{}0{}_{}'.format(dep, left_or_right, tag))
    return co_x, co_y

def Mi_coor2(Im_x, Im_y, co_x, co_y):
    """
        使用3D坐标+标定参数，将图像坐标映射到镜面坐标
    """
    Mi_x = np.dot(
        np.array([np.array([1]*98), Im_x, Im_y]).T,
        co_x
        ).astype(int)
    Mi_y = np.dot(
        np.array([np.array([1]*98), Im_x, Im_y]).T,
        co_y
        ).astype(int)
    return Mi_x, Mi_y
def Mi_coor(Ca_coor, co_x, co_y):
    """
        使用3D坐标+标定参数，将图像坐标映射到镜面坐标
    """
    Ca_x, Ca_y, Ca_z = Ca_coor[:, 0], Ca_coor[:, 1], Ca_coor[:, 2]
    px = np.poly1d(co_x)
    Mi_x = px(Ca_x).astype(int)
    Mi_y = np.dot(
        np.array([np.array([1]*98), Ca_y, Ca_z]).T,
        co_y
        ).astype(int)
    return Mi_x, Mi_y

def soft_weight(Im_x, Im_y):
    '''
        使用softmax计算5个mask各自的权重
    '''
    im = np.array([Im_x[79], Im_y[79]])  # 人中
    dis = np.sum(np.square(Im_anchor - im), axis=1)
    bias = 0.001  # 防止dis==0时Error
    s = 1 / (dis+bias)
    soft = np.exp(s) / np.sum(np.exp(s))
    return soft

def masks(Im_x, Im_y, dep, Ca_coor):
    '''
        n个mask加权求和得到最终mask位置，n==5
    '''
    Mi_xs, Mi_ys = np.zeros([5, 98]), np.zeros([5, 98])  # 初始化
    for i in range(len(Im_anchor)):  # 选取不同标定参数，计算多个mask坐标
        co_x, co_y = fun_co(dep, i)  # 得到标定参数
        Mi_xs[i], Mi_ys[i] = Mi_coor(Ca_coor, co_x, co_y)  # 计算多个mask横纵坐标并保存到2D矩阵中
    soft = soft_weight(Im_x, Im_y)  # 计算不同mask的权重概率
    Mi_x = np.sum(soft*Mi_xs, axis=0)  # 多个mask加权求和
    Mi_y = np.sum(soft*Mi_ys, axis=0)  # 多个mask加权求和
    return Mi_x, Mi_y

def Mi_coor_smooth_vertical(Ca_coor, dep):
    """
        距离可变
        通过插值，平滑化深度改变时的mask大小变化幅度
        以dep=0.43为例，分别计算dep=0.4 & dep=0.5时mask的镜面坐标Mi_coor
        再以dep=0.4的坐标为基础，向dep=0.5的坐标偏移30%，其结果作为最终Mi_coor
    """
    dep_down = int(dep*10)/10.
    co_x, co_y = fun_co(dep_down, tag=0)
    Mi_x_down, Mi_y_down = Mi_coor(Ca_coor, co_x, co_y)

    dep_up = int(dep*10)/10. + 0.1
    co_x, co_y = fun_co(dep_up, tag=0)
    Mi_x_up, Mi_y_up = Mi_coor(Ca_coor, co_x, co_y)

    fine_ratio = int(dep*100)%10 / 10
    Mi_x = (Mi_x_down + (Mi_x_up-Mi_x_down)*fine_ratio).astype(int)
    Mi_y = (Mi_y_down + (Mi_y_up-Mi_y_down)*fine_ratio).astype(int)
    return Mi_x, Mi_y

def Mi_coor_smooth_parallel(Ca_coor, Im_x, Im_y, dep):
    '''
        固定距离，多组参数
    '''
    Mi_x, Mi_y = masks(Im_x, Im_y, 0.4, Ca_coor)
    return Mi_x, Mi_y