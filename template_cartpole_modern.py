import gymnasium as gym
import numpy as np

lp = 0.5
mp = 0.1
mk = 1.0
mt = mp + mk
g = 9.8
K = np.array([-10.0000,  -24.8801, -265.0991, -119.4685])


# get environment
env = gym.make('CartPole-v1', render_mode = 'human')
(obs, info) = env.reset(seed=1)
reward_total = 0


# ADD SOMETHING HERE


def apply_state_controller(K, x):
    # feedback controller
    # MODIFY THIS PARTS
    u = -np.dot(K, x)  # u = -Kx
    if u > 0:
        return 1, u  # if force_dem > 0 -> move cart right
    else:
        return 0, u  # if force_dem <= 0 -> move cart left


for i in range(1000):
    env.render()

    # get force direction (action) and force value (force)

    # MODIFY THIS PART
    action, force = apply_state_controller(K, obs)
    print(force, "----" , action)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))

    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _, _= env.step(action)

    reward_total = reward_total + reward
    if reward_total > 300:
        print(reward_total)
        u = np.array([force])
        print(u)
        break
    if done:
        print(f'Terminated after {i + 1} iterations.')
        print(reward_total)
        break

env.close()