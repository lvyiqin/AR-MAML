
"""
This is for classic Acrobot task.
"""

import random
import numpy as np
from numpy import sin, cos, pi
import torch

from gym import core, spaces
from gym.utils import seeding



# SOURCE: https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class AcrobotEnv(core.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    
    LINK_COM_POS_1 = 0.5  #: position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: position of the center of mass of link 2
    LINK_MOI = 1.  # moments of inertia for both links

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = [-1., 0., +1]

    torque_noise_max = 0.

    # use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self, m_1 = 1.0, m_2 = 1.0):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()
        
        self.LINK_MASS_1 = m_1  #: [kg] mass of link 1
        self.LINK_MASS_2 = m_2  #: [kg] mass of link 2
        self.env_name = 'acrobot'

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def step(self, a):
        s = self.state
        torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4] 

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):

    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):

    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
    

def check_done_state_torch(obs):

    cos_s0=obs[...,0]
    s0=torch.atan2(obs[...,1],obs[...,0])
    s1=torch.atan2(obs[...,3],obs[...,2])
    terminal_value=-cos_s0-torch.cos(s0+s1)
    check_done=(terminal_value>torch.tensor(1.0).cuda())
    
    return check_done
    
    
def acrobot_reward_torch(obs):

    check_done=check_done_state_torch(obs)
    if check_done:
        return torch.tensor(0.0,requires_grad=True)
    else:
        return torch.tensor(-1.0,requires_grad=True)
    

def check_done_state_numpy(obs):

    cos_s0=obs[...,0]
    s0=np.arctan2(obs[...,1],obs[...,0])
    s1=np.arctan2(obs[...,3],obs[...,2])
    terminal_value=-cos_s0-np.cos(s0+s1)
    check_done=bool(terminal_value>1.0)
    
    return check_done
    
       
def acrobot_reward_numpy(obs):

    if torch.is_tensor(obs):
        obs=obs.cpu().numpy()
    check_done=check_done_state_numpy(obs)
    
    if check_done_state_numpy(obs):
        return 0.0
    else:
        return -1.0


##########################################################################################################################
    # This part is to introduce the task sampling method.
##########################################################################################################################


# def sample_batch_envs(m_1_list, m_2_list):
#     env_list = []
#     for m_1_value in m_1_list:
#         for m_2_value in m_2_list:
#             acrobot_env=AcrobotEnv(m_1=m_1_value,m_2=m_2_value)
#             env_list.append(acrobot_env)

#     return env_list

def sample_batch_envs(m_1_list, m_2_list):
    env_list = []
    for i in range(m_1_list.shape[0]):
        acrobot_env=AcrobotEnv(m_1=m_1_list[i],m_2=m_2_list[i])
        env_list.append(acrobot_env)

    return env_list



