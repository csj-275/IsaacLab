# lqr控制
import mujoco
import mujoco.viewer
from math import sin, cos, pi
import numpy as np
from scipy.integrate import solve_ivp
import control
import time

def normalize_angle(angle):
    """将角度归一化到 (-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def lqr_control(dt):
    '''
    动量轮古田摆动力学模型
    输入 q(np.array) dq(np.array) tau(float)
    输出 ddq(np.array)
    '''

    alpha = pi/6
    J0 = 0.001093753
    J1 = 1.6547*10**(-5) # 38.677*10**(-6)
    J2 = 0.001011296 #4991*10*(-6)# 0.0010112
    
    L1 = 0.052
    L2 = 0.137

    l1 = 0.009120
    l2 = 0.035022

    m1 = 0.03847
    m2 = 0.5886
    m0 = 0.518
    g = 9.81

    b1 = 2*J0
    b2 = J0 + J1 + m1*l1**2 + (m0+m2)*L1**2
    b3 = m2*l2**2 + m0*L2**2
    b4 = J0 * sin(alpha)**2
    b5 = J0*(1+cos(alpha)**2) + J2 + m2*l2**2 + m0*L2**2 + J0*cos(alpha)**2
    b6 = L1*(m0*L2 + m2*l2) - J0*sin(2*alpha)
    b7 = 2*J0*sin(alpha)
    b8 = J0*cos(alpha)
    b9 = m2*g*l2 + m0*g*L2

    detM = b1*b6+b2*b8**2+b4*b8**2+b5*b7**2-b1*b2*b5-b1*b4*b5+2*b6*b7*b8
    a43 = -b9*(b2*b8+b4*b8+b6*b7)/detM
    a53 = -b9*(b1*b6+b7*b8)/detM
    a63 = b9*(b1*b2+b1*b4-b7**2)/detM
    b4 = (b6**2-b4*b5-b2*b5)/detM
    b5 = -(b5*b7+b6*b8)/detM
    b6 = (b2*b8+b4*b8+b6*b7)/detM

    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, a43, 0, 0, 0],
                  [0, 0, a53, 0, 0, 0],
                  [0, 0, a63, 0, 0, 0]])
    B = np.array([[0],
                  [0],
                  [0],
                  [b4],
                  [b5],
                  [b6]])
    A = np.eye(6) + A * dt
    B = B * dt
    Q = np.diag([-0.01, -10, 75, -1, -0.5, 2])
    R = 0.001
    K = control.dlqr(A,B,Q,R)[0]
    K = -K
    return K


def write_pos(pos, filename):
    '''
    功能：写入数据
    参数: pos(list) filename(str)
    '''
    list1 = ( ",".join(repr(e) for e in pos))
    with open(filename,'a') as f:
        f.write('\n'+str(list1))

def clear_record():
    '''
    功能：清空文件数据
    '''
    filename1 = 'record_mujoco.txt'
    filename2 = 'record_model.txt'
    open(filename1, 'w').close()
    open(filename2, 'w').close()

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
data.qpos = np.array([1.5, 0, 0])  # 初始化位置 dq1 dq2 dq0

dt = 0.001 # 仿真步长

# 数据文件
filename1 = 'record_mujoco.txt'
filename2 = 'record_model.txt'
# 清除数据
clear_record()
# K = lqr_control(dt)
# q0 q1 q2
K = np.array([[0.0, -30, 75.0, 1.5, -1.5, 2]])  # 极点配置 可用
# K = np.array([[0.0, 0, 75.0, 0.0, 0, 2]])  # 极点配置 可用

# 循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 获取信息
        q1, q2, q0 = data.qpos.copy().tolist()
        dq1, dq2, dq0 = data.qvel.copy().tolist()
        # q1 = normalize_angle(q1)
        # q2 = normalize_angle(q2)
        q = np.array([q0,q1,q2]).reshape(3,-1)
        dq = np.array([dq0,dq1,dq2]).reshape(3,-1)
        print('=====================')
        print('误差：',q2-pi)
        print(f'当前状态:{q0:.2f},{q1:.2f},{q2:.2f},{dq0:.2f},{dq1:.2f},{dq2:.2f}')
        tau = (-K @ np.array([q0, q1, (q2-pi), dq0, dq1, dq2]).reshape(6, -1))[0][0]
        print(f'计算力矩:{tau:.2f}')
        tau = np.clip(tau, -3, 3)
        data.ctrl = tau # 力矩输入
        print(f'控制力矩:{tau:.2f}')
        # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
        # 写入数据
        write_pos(np.concatenate((q, dq)).reshape(1, -1).tolist()[0], filename1)
