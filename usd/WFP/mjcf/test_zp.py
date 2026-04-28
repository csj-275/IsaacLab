import mujoco
import mujoco.viewer
from math import sin,cos,pi
import numpy as np
from scipy.integrate import solve_ivp


def dynamic_model(q, dq, tau):
    '''
    动量轮古田摆动力学模型
    输入 q(np.array) dq(np.array) tau(float)
    输出 ddq(np.array)
    '''
    _, _, q2 = q
    _, dq1, dq2 = dq
    q = q.reshape(3,1)
    q = np.clip(q, -np.pi/2, np.pi/2)
    dq = dq.reshape(3,1)
    alpha = pi/6
    J0 = 0.00109375279298174
    J1 = 1.6547*10**(-5) # 38.677*10**(-6)
    J2 = 0.0010112
    L1 = 0.040
    L2 = 0.14
    l1 = 0.00912
    l2 = 0.035
    m1 = 0.038
    m2 = 0.589
    m0 = 0.518
    g = 9.81

    L2 = 1.107
    theta = pi/6
    b1 = 2*J0

    b1 = 2*J0
    b2 = J0 + J1 + m1*l1**2 + (m0+m2)*L1**2
    b3 = m2*l2**2 + m0*L2**2
    
    b5 = J0*(1+cos(alpha)**2) + J2 + m2*l2**2 + m0*L2**2 + J0*cos(alpha)**2
    b6 = L1*(m0*L2 + m2*l2) - J0*sin(2*alpha)
    
    b8 = J0*cos(alpha)
    b9 = m2*g*l2 + m0*g*L2

    b4 = J0 * sin(alpha)**2
    b7 = 2*J0*sin(alpha)
    mb = 1.107
    L0 = 0.0395
    L1 = 0.153
    Ib = 16591*10**(-6)
    M = np.array([[2*J0, -2*J0*sin(alpha)*cos(q2), 2*J0*cos(alpha)],
                  [-2*J0*sin(alpha)*cos(q2), L0**2*mb-J0/4*(-5+cos(2*alpha)-2*cos(2*q2)*sin(alpha)**2), cos(q2)*(mb*L0*L1-J0*sin(alpha)*cos(alpha))],
                  [2*J0*cos(alpha), (L0*L1*mb-J0*sin(alpha)*cos(alpha))*cos(q2), (2*Ib+2*mb*L1**2+J0*(3+cos(2*alpha)))/2]])

    C = np.array([[0, b7*sin(q2)*dq2, 0],
                  [b7*sin(q2)*dq2, (mb*L1**2-b4)*sin(2*q2)*dq2, (J0*sin(alpha)*cos(alpha)-mb*L0*L1)*sin(q2)*dq2],
                  [-b7*sin(q2)*dq1, (-mb*L1**2+b4)*sin(2*q2)*dq1/2, 0]])
    G = np.array([[0],
                  [0],
                  [-mb*g*L1*sin(q2)]])
    F = np.array([[tau],
                  [0],
                  [0]])
    
    ddq = np.linalg.inv(M) @ (-C @ dq - G) + np.linalg.inv(M) @ F
    return ddq

def rhs_symbolic(t, y, tau):
    q, dq = y[:3], y[3:]               # 3 个广义坐标
    ddq = dynamic_model(q, dq, tau).squeeze()
    return np.concatenate([dq, ddq])

def rk4_step(f, t, y, h, *args):
    """
    经典 4 阶 RK4 单步
    f : rhs(t, y, *args)  -> dy/dt
    """
    k1 = f(t,         y,            *args)
    k2 = f(t + h/2.0, y + h/2*k1,   *args)
    k3 = f(t + h/2.0, y + h/2*k2,   *args)
    k4 = f(t + h,     y + h*k3,     *args)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6.0

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

model = mujoco.MjModel.from_xml_path("mjcf/scene.xml")
data = mujoco.MjData(model)

dt = 0.001 # 仿真步长
# 数据记录
T, Q, Qd, Qdd = [], [], [], []
Q_sym, Qd_sym, Qdd_sym = [], [], []
# 仿真准备
t_sym = 0.0
y_sym = np.zeros(6)

# 数据文件
filename1 = 'record_mujoco.txt'
filename2 = 'record_model.txt'
# 清除数据
clear_record()

# 循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        t = data.time
        tau = 0.5*sin(t)
        data.ctrl = tau # 力矩输入

        # 直接获取信息
        # q = data.qpos.copy()
        # dq = data.qvel.copy()
        # 传感器获取信息
        q1 = data.sensor("q1").data[0]
        q2 = data.sensor("q2").data[0]
        q0 = data.sensor("q3").data[0]
        dq1 = data.sensor("dq1").data[0]
        dq2 = data.sensor("dq2").data[0]
        dq0 = data.sensor("dq3").data[0]
        q = np.array([q0,q1,q2]).reshape(3,-1)
        dq = np.array([dq0,dq1,dq2]).reshape(3,-1)

        print('-------')
        # 数值积分
        y_sample = np.concatenate((q, dq)).reshape(-1,6)[0]
        print(y_sample)
        # sol = solve_ivp(rhs_symbolic,
        #                 [t_sym, t_sym + dt],
        #                 y_sym,
        #                 method='RK45',
        #                 t_eval=[t_sym + dt],
        #                 rtol=1e-10, atol=1e-10)
        y_sym = rk4_step(rhs_symbolic, t_sym, y_sym, dt, tau)
        t_sym += dt
        
        # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()

        # 写入数据
        write_pos(np.concatenate((q, dq)).reshape(1, -1).tolist()[0], filename1)
        write_pos(y_sym, filename2)