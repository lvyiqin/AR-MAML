import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import torch


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, m = 1.0, l = 1.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = 10.0
        self.m = m
        self.l = l
        self.viewer = None
        self.env_name = 'pendulum'

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def compute_cost(self,action):
        th, thdot = self.state
        cost = pendulum_cost_numpy(th, thdot, action)
        return cost    

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        cost = self.compute_cost(u)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -cost, np.array(0.0), {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot]) #note state space is 3 because sin and cos can dertermine the location

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def pendulum_cost_numpy(th, thdot, action):
    # th := theta
    th, thdot, action = np.array(th), np.array(thdot), np.array(action)
    u = np.clip(action, -2.0, 2.0) #normalize the action to the plausible domain
    cost = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    
    return cost


def pendulum_reward_numpy(obs, action):
    # th := theta
    th = np.arctan2(obs[...,1], obs[...,0])
    thdot = obs[...,2]
    th, thdot, action = np.array(th), np.array(thdot), np.array(action)
    u = np.clip(action, -2.0, 2.0) #normalize the action to the plausible domain
    cost = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    
    return -cost


def pendulum_cost_torch(state, action):
    '''
    Torch version used for policy learning in dm, not for the real env.
    Input should be obs=[cos theta, sin theta, delta theta] and action in batch
    Note state is 2-dim [theta, delta theta] but obs is 3-dim [cos theta, sin theta, delta theta]
    '''
    # th := theta
    state, action = np.array(state.detach().cpu()), np.array(action.detach().cpu())
    th = state[...,0] # (batch_size,) to compute theta from cos and sin value
    thdot = state[...,1]
    u = np.clip(action, -2.0, 2.0) #normalize the action to the plausible domain
    cost = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    cost = torch.tensor(cost)
    
    return cost


def pendulum_reward_torch(state, action):
    '''
    Torch version used for policy learning in dm, not for the real env.
    Input should be obs=[cos theta, sin theta, delta theta] and action in batch
    Note state is 2-dim [theta, delta theta] but obs is 3-dim [cos theta, sin theta, delta theta]
    '''
    # th := theta
    state, action = np.array(state.detach().cpu()), np.array(action.detach().cpu())
    th = state[...,0] # (batch_size,) to compute theta from cos and sin value
    thdot = state[...,1]
    u = np.clip(action, -2.0, 2.0) #normalize the action to the plausible domain
    cost = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    cost = torch.tensor(cost)
    
    return -cost



##########################################################################################################################
    # This part is to introduce the task sampling method.
##########################################################################################################################


# def sample_batch_envs(m_list, l_list):
    
#     env_list = []
#     for m_value in m_list:
#         for l_value in l_list:
#             pendulum_env=PendulumEnv(m=m_value,l=l_value)
#             env_list.append(pendulum_env)

#     return env_list


def sample_batch_envs(m_list, l_list):
    env_list = []
    for i in range(m_list.shape[0]):
        pendulum_env=PendulumEnv(m=m_list[i],l=l_list[i])
        env_list.append(pendulum_env)

    return env_list
