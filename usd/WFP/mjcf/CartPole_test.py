import mujoco
import mujoco.viewer
from math import sin, cos, pi, sqrt
import numpy as np
from scipy.integrate import solve_ivp


def dynamic_model(q, dq, tau):
    '''
    动量轮古田摆动力学模型
    输入 q(np.array) dq(np.array) tau(float)
    输出 ddq(np.array)
    '''
    q1, _ = q
    dq1, dq2 = dq
    q = q.reshape(2,1)
    q = np.clip(q, -np.pi, np.pi)
    dq = dq.reshape(2,1)
    dq = np.clip(dq, -np.pi, np.pi)
    m1 = 2.0622
    m2 = 0.51812
    l1 = 0.173741
    L1 = 0.334
    m_hat = m1*l1 + m2*L1
    # I1 = 84215793 * 10**(-9)
    I1=  21964603 * 10**(-9)
    # print(I11)
    I2 = 0.0021789
    g = 9.81

    M = np.array([[m1*l1**2+m2*L1**2+I1+I2, I2],
                  [I2, I2]])
    
    G = np.array([[-m_hat*g*sin(q1)],
                  [0]])
    
    F = np.array([[0],
                  [tau]])
    d11 = M[0,0] 
    d12 = M[0,1]
    d21 = M[1,0]
    d22 = M[1,1]
    detD = np.linalg.det(M)
    g = m_hat*g*sin(q1)
    
    ddq1 = d22*g/detD - d12/detD*tau
    ddq2 = -d21*g/detD + d11/detD*tau
    ddq = np.array([ddq1,ddq2]).reshape(2,1)
    # ddq = np.dot(np.linalg.inv(M),-G) + np.dot(np.linalg.inv(M),F)
    # ddq = np.linalg.inv(M) @ (-C @ dq - G) + np.linalg.inv(M) @ F
    return ddq

def rhs_symbolic(t, y, tau):
    q, dq = y[:2], y[2:]               # 2 个广义坐标
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

model = mujoco.MjModel.from_xml_path("mjcf/cartpole.xml")
data = mujoco.MjData(model)

dt = 0.001 # 仿真步长

# 仿真准备
t_sym = 0.0
y_sym = np.zeros(4)

# 数据文件
filename1 = 'record_mujoco.txt'
filename2 = 'record_model.txt'
# 清除数据
clear_record()

# 初始化激励参数
freqs = np.logspace(-1, 1, 20)  # 0.1Hz到10Hz对数均匀分布
amps = 30 * np.ones_like(freqs) / len(freqs)


# 循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = data.time
        tau = 0.1*sin(5*t)
        # tau = sum(a * np.sin(2*np.pi*f*t) for f, a in zip(freqs, amps))
        print(tau)
        data.ctrl = tau # 力矩输入

        # # 直接获取信息
        # q1, q2 = data.qpos.copy().tolist()
        # dq1, dq2 = data.qvel.copy().tolist()
        # ddq1, ddq2 = data.qacc.copy().tolist()
        # print(q,dq)

        # 传感器获取信息
        # q1 = data.sensor("q1").data[0]
        # q2 = data.sensor("q2").data[0]
        # q0 = data.sensor("q3").data[0]
        # dq1 = data.sensor("dq1").data[0]
        # dq2 = data.sensor("dq2").data[0]
        # dq0 = data.sensor("dq3").data[0]
        
        # q = np.array([q1,q2]).reshape(2,-1)
        # dq = np.array([dq1,dq2]).reshape(2,-1)
        # ddq = np.array([ddq1,ddq2]).reshape(2,-1)
        
        # print('-------')
        # # 数值积分
        # y_sample = np.concatenate((q, dq)).reshape(-1,4)[0]
        # # sol = solve_ivp(rhs_symbolic,
        # #                 [t_sym, t_sym + dt],
        # #                 y_sym,
        # #                 method='RK45',
        # #                 t_eval=[t_sym + dt],
        # #                 rtol=1e-10, atol=1e-10)
        # y_sym = rk4_step(rhs_symbolic, t_sym, y_sym, dt, tau)
        # ddq_cal = dynamic_model(y_sym[:2],y_sym[2:], tau).squeeze()
        # y_record = np.concatenate([y_sym,ddq_cal])
        # t_sym += dt
        
        # # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()

        # # 写入数据
        # write_pos(np.concatenate((q, dq, ddq)).reshape(1, -1).tolist()[0], filename1)
        # write_pos(y_record, filename2)