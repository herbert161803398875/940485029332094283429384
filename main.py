import gymnasium as gym
import numpy as np
import control
import time

# Constants
lp = 0.5
mp = 0.1
mk = 1.0
mt = mp + mk
g = 9.8
# default
# K = np.array([-10.0000,  -24.8801, -265.0991, -119.4685])
# L = np.array([[203.50869858, 0], [486.40757794, 0], [4488.51459357, 0],[10163.6454345, 0]])  # You need to define L according to your system dynamics and compensator design
# rekor baru settling time 0.74, ov 59%
K = np.array([-14.1421, -25.1912, -229.9703, -65.7432])

L = np.array([[14.2531, -0.0164],
              [1.5762, -0.8901],
              [-0.0164, 141.8605],
              [-1.6691, 62.1956]])
#  KURLEB LEBIH BAIK DI OV TAPI PEAKTIME MASIH JELEK 7.66 S, K ATAS PLUS L BAWAH DAPET OV 51% DAN PEAK TIME 1.9S
# K = np.array([-14.1421, -22.9291, -192.9457, -37.5354])
# #
# L = np.array([[14.1835, -0.0339],
#               [0.5862, -1.1302],
#               [-0.0339, 141.6705],
#               [-4.1495, 35.2666]]#    #               )


# K = np.array([-14.1421, -25.1403, -229.7745, -65.6945])

# L = np.array([[14.1907, -0.0187],
#               [0.6888, -0.9166],
#               [-0.0187, 141.8604],
#               [-2.0039, 62.1826]])
# K = np.array([-14.1421, -21.9291, -162.9457, -37.5354])
# L = np.array([[14.2531, -0.0164],
#                [1.5762, -0.8901],
#                [-0.0164, 141.8605],
#                [-1.6691, 62.1956]])


# K = np.array([-39.2232, -45.7260, -271.9911, -75.4347])
#
# L = np.array([[39.2404, -0.0107],
#               [0.6742, -0.9613],
#               [-0.0107, 124.4938],
#               [-0.7833, 57.0432]])
A = np.array([[0, 1, 0, 0],
              [0, 0, -0.7171, 0],
              [0, 0, 0, 1],
              [0, 0, 15.0419, 0]])
B = np.array([[0],
              [0.9756],
              [0],
              [-1.3953]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],
              [0]])
Ar = np.array([[0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, -0.7171, 0],
               [0, 0, 15.0419, 0]])

Br = np.array([[0],
               [0],
               [0.9756],
               [-1.3953]])
Aaa = np.array([[0, 1],
                [0, 0]])
Aab = np.array([[0, 0],
                [0, 1]])
Aba = np.array([[0, 0],
                [0, 0]])
Abb = np.array([[-0.7171, 0],
                [15.0419, 0]])

dt = 0.001
delta_t = 0.02
# Initialize states
x_hat_0 = np.array([0.05,
                    0,
                    0.05,
                    0])

# Initialize environment
env = gym.make('CartPole-v1', render_mode='human')
(obs, info) = env.reset(seed=1)
state = obs
reward_total = 0


# add something here
def apply_compensator(A, B, C, force, L, x_hat_0, state):
    # State feedback
    u = force
    x = state
    x_hat = x_hat_0
    # State estimator
    # x = x[:, np.newaxis]
    x_hat_dot = A @ x_hat + B @ u + L @ (C @ x - C @ x_hat)

    return x_hat_dot


def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -np.dot(K, x)  # u = -Kx
    if u > 0:
        return 1, u  # if force_dem > 0 -> move cart right
    else:
        return 0, u  # if force_dem <= 0 -> move cart left


# Copy this for scoring
t_array = []
theta_array = []
force_array = []
T = 0

for i in range(1000):
    env.render()
    # get the output only:
    y = np.array([[state[0]], [state[2]]])

    # Apply compensator
    action, force = apply_state_controller(K, x_hat_0)  # butuh x_hat, dapet u sama action (1 atau 0)
    u = np.resize(force, (1, 1))
    print("u:", u)
    u = np.array([force])

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))  # dapet abs(u)
    force = np.float32(np.clip(force, -10, 10))
    x_hat_dot = apply_compensator(A, B, C, u, L, x_hat_0, state)
    x_hat_0 = state + x_hat_dot * dt

    # env.env.state = x_hat_0.reshape(-1)
    state, reward, done, _, _ = env.step(action)

    # Update states
    print(state, "<-state")
    print(x_hat_dot, "<-x_hat_dot")
    print(x_hat_0, "<-x_hat_0")
    print("sudah berada pada iterasi ke-", i)
    # Determine action based on control input u
    # change magnitude of the applied force in CartPole
    env.force_mag = abs_force
    # apply action
    reward_total = reward_total + reward

    # Copy this for scoring
    t_array.append(T)
    theta_array.append(state[2])
    T = T + delta_t
    time.sleep(delta_t)

    if done:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break
    if i > 998:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break
env.close()


## Calculate performance
def perform_calc(y_array, t_array):
    y_max = np.amax(y_array)
    y_min = np.amin(y_array)
    y_ss = np.average(y_array[-101:-1] - y_min)

    # Overshoot (in Percentage)
    OV = ((y_max - y_min) / y_ss - 1) * 100
    if (OV < 1e-6): OV = 0

    # Peak Time
    peak_idx = np.argmax(y_array)
    peak_time = t_array[peak_idx]

    # Settling Time
    TH = 0.02  # 2 % Criterion
    y_set = 0
    n_set = 0
    n_prev = 0
    datapoints = len(t_array)
    for n in range(0, datapoints):
        if (np.abs(y_array[n, :] - y_min - y_ss) <= TH * y_ss) and (n_prev == 0):
            y_set = y_array[n, :]
            n_set = n
            n_prev = 1
        elif (np.abs(y_array[n, :] - y_min - y_ss) > TH * y_ss):
            n_prev = 0
    settling_time = t_array[n_set]

    return OV, peak_time, settling_time, y_min, y_max, y_ss


theta_array = np.array(theta_array).reshape((len(theta_array), 1))
OV, peak_time, settling_time, y_min, y_max, y_ss = perform_calc(theta_array, np.array(t_array))
print('result is.....')
print('OV =', OV)
print('peak time = ', peak_time)
print('settling_time = ', settling_time)
print('y_min = ', y_min)
print('y_max = ', y_max)
print('y_ss =', y_ss)
