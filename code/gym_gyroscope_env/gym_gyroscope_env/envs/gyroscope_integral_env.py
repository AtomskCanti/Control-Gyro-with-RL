import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp

class GyroscopeIntegralEnv(gym.Env):


    """
    GyroscopeEnv is a double gimbal control moment gyroscope (DGCMG) with 2 input voltage u1 and u2
    on the two gimbals, and disk speed assumed constant (parameter w). Simulation is based on the
    Quanser 3-DOF gyroscope setup.


    **STATE:**
    The state consists of the angle and angular speed of the outer red gimbal (theta = x1, thetadot = x2),
    the angle and angular speed of the inner blue gimbal (phi = x3, phidot = x4), the reference
    for tracking on theta and phi (x1_ref and x3_ref), and the disk speed (disk speed = w):

    state = [x1, x2, x3, x4, x1_ref, x3_ref, w]

    **OBSERVATION:**
    The observation consists of the state where the angles have been replaced with their cosine and sine to
    prevent the discontinuity at -pi/pi, the observation space is thus larger than the state space.

    observation =  [cos(x1), sin(x1), x2, cos(x3), sin(x3), x4, x1_ref, x3_ref, w] (9 dimensions)

    **ACTIONS:**
    The actions are the input voltage to create the red and blue gimbal torque (red voltage = u1, blue voltage = u2),
    and are continuous in a range of -10 and 10V:

    action = [u1,u2]

    """


    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        # Inertias in Kg*m2
        self.Jbx1 = 0.0019
        self.Jbx2 = 0.0008
        self.Jbx3 = 0.0012
        self.Jrx1 = 0.0179
        self.Jdx1 = 0.0028
        self.Jdx3 = 0.0056

        # Combined inertias
        self.J1 = self.Jbx1 - self.Jbx3 + self.Jdx1 - self.Jdx3
        self.J2 = self.Jbx1 + self.Jdx1 + self.Jrx1
        self.J3 = self.Jbx2 + self.Jdx1

        # Motor constants
        self.Kamp = 0.5 # A/V
        self.Ktorque = 0.0704 # Nm/A
        self.eff = 0.86
        self.nRed = 1.5
        self.nBlue = 1
        self.KtotRed = self.Kamp*self.Ktorque*self.eff*self.nRed
        self.KtotBlue = self.Kamp*self.Ktorque*self.eff*self.nBlue

        # Disk speed
        self.w = 0

        # Integrals
        self.diff_x1_int = 0
        self.diff_x3_int = 0

        # Time step in s
        self.dt = 0.05
        self.eval_per_dt = 5

        # Reward function by default for SpiningUp functions that cannot init using args_int
        self.reward_func = norm_reward
        self.reward_args = {'k': 0.05, 'qx2' : 0.01, 'qx4' : 0.01, 'pu1':0.5,'pu2':0.5}


        # Episode length and current episode
        self.ep_len = 110 # default
        self.ep_curr = 0

        # Action space
        self.maxVoltage = 10 # V
        self.highAct = np.array([1,1])
        self.action_space =  spaces.Box(low = -self.highAct, high = self.highAct, dtype=np.float32)

        # State and observation space
        self.maxSpeed = 100 * 2 * np.pi / 60
        self.maxAngle = np.pi
        self.maxdiskSpeed = 300 * 2 * np.pi / 60
        self.highState = np.array([self.maxAngle,self.maxSpeed,self.maxAngle,self.maxSpeed,self.maxAngle,self.maxAngle,self.maxdiskSpeed,0,0])
        self.highObs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf])
        self.lowObs = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,-np.inf, -np.inf])
        self.observation_space = spaces.Box(low = self.lowObs, high = self.highObs, dtype=np.float32)

        # Seed for random number generation
        self.seed()

        self.viewer = None

    # method used to try different variants of the env
    def args_init(self, reward_type, reward_args, ep_len, is_noise=False):
        # Reward function type
        reward_dict = {
            'Quadratic': quad_reward,
            'Absolute': abs_reward,
            'Normalized': norm_reward,
            'Quadratic with ending penalty': quad_end_pen_reward,
            'Quadratic with penalty': quad_pen_reward,
            'Quadratic with exponential': quad_exp_reward,
            'Quadratic with bonus':quad_bon_reward
        }
        self.reward_func = reward_dict[reward_type]
        self.reward_args = reward_args

        # Episode length
        self.ep_len = ep_len

        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        x1, x2, x3, x4, x1_ref, x3_ref, _,_,_ = self.state
        u1,u2 = u

        # Increment episode
        self.ep_curr += 1

        # For quad_end_pen_reward, check if terminal state is reached
        if self.reward_func == quad_end_pen_reward and self.ep_curr == self.ep_len:
            self.reward_args['end_horizon'] = 1

        # Simulate
        results = solve_ivp(fun = dxdt, t_span = (0, self.dt), y0 = [x1,x2,x3,x4], method='RK45',  t_eval = np.linspace(0,self.dt,self.eval_per_dt), args=(self.maxVoltage*u1,self.maxVoltage*u2,self))

        # For rendering
        x1_eval = (results.y[0])
        x2_eval = np.clip(results.y[1],-self.maxSpeed,self.maxSpeed)
        x3_eval = (results.y[2])
        x4_eval = np.clip(results.y[3],-self.maxSpeed,self.maxSpeed)

        # For state
        x1 = x1_eval[-1]
        x2 = x2_eval[-1]
        x3 = x3_eval[-1]
        x4 = x4_eval[-1]

        self.state = np.asarray([x1,x2,x3,x4,x1_ref,x3_ref,self.w,self.diff_x1_int,self.diff_x3_int])

        # Angle error (normalized between pi and -pi this time to get smallest distance)
        diff_x1 = angle_normalize_pi(x1 - x1_ref)
        diff_x3 = angle_normalize_pi(x3 - x3_ref)

        # Integrals
        self.diff_x1_int += diff_x1
        self.diff_x3_int += diff_x3

        # Reward
        reward = self.reward_func(diff_x1, diff_x3, x2, x4, u1, u2, **self.reward_args)

        return self._get_obs(), reward, False,{'state':self.state,'x1_eval':x1_eval,'x2_eval':x2_eval,'x3_eval':x3_eval,'x4_eval':x4_eval,'x1_ref_eval':np.full(self.eval_per_dt,x1_ref),'x3_ref_eval':np.full(self.eval_per_dt,x3_ref)}

    def _get_obs(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), s[1]/self.maxSpeed, np.cos(s[2]), np.sin(s[2]), s[3]/self.maxSpeed, s[4]/self.maxAngle, s[5]/self.maxAngle,s[6]/self.maxdiskSpeed,np.sign(self.diff_x1_int)/(1+abs(self.diff_x1_int)),np.sign(self.diff_x3_int)/(1+abs(self.diff_x3_int))])


    def reset(self):
        self.state = self.np_random.uniform(low=-self.highState, high=self.highState)
        self.state[-2:] = [0,0] # Set integrals to 0
        self.diff_x1_int = 0
        self.diff_x3_int = 0
        self.w = self.state[-3]
        self.ep_curr = 0
        return self._get_obs()

    # cannot be implemented in reset() in gym API, need to have separate method
    def reset2state(self, state):
        self.state = state
        self.state[-2:] = [0,0] # Set integrals to 0
        self.diff_x1_int = 0
        self.diff_x3_int = 0
        self.w = self.state[-3]
        self.ep_curr = 0
        return self._get_obs()

    def access_members(self):
        return self.dt,self.eval_per_dt,self.maxVoltage

    def access_state(self):
        return self.state


    def render(self, mode='human'):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def dxdt(t, x, u1, u2, gyro):

        # Rewrite constants shorter
        J1 = gyro.J1
        J2 = gyro.J2
        J3 = gyro.J3
        Jdx3 = gyro.Jdx3
        KtotRed = gyro.KtotRed
        KtotBlue = gyro.KtotBlue
        w = gyro.w

        # Convert input voltage to input torque
        u1,u2 = KtotRed*u1, KtotBlue*u2

        # Equations of motion
        dx_dt = [0, 0, 0, 0]
        dx_dt[0] = x[1]
        dx_dt[1] = (u1+J1*np.sin(2*x[2])*x[1]*x[3]-Jdx3*np.cos(x[2])*x[3]*w)/(J2 + J1*np.power(np.sin(x[2]),2))
        dx_dt[2] = x[3]
        dx_dt[3] = (u2 - J1*np.cos(x[2])*np.sin(x[2])*np.power(x[1],2)+Jdx3*np.cos(x[2])*x[1]*w)/J3
        return dx_dt

def quad_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0):
    return -(qx1*(diff_x1**2) + qx3*(diff_x3**2) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2))

def quad_exp_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0,eax1 = 10,ebx1=10,eax3 = 10,ebx3=10):
    return -(qx1*(diff_x1**2) + qx3*(diff_x3**2) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2) + eax1*(1-np.exp(-ebx1*(diff_x1**2))) + eax3*(1-np.exp(-ebx3*(diff_x3**2)))  )

def abs_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0):
    return -(qx1*abs(diff_x1) + qx3*abs(diff_x3) + qx2*abs(x2) + qx4*abs(x4) + pu1*abs(u1) + pu2*abs(u2))

def norm_reward(diff_x1, diff_x3, x2, x4, u1, u2, k = 0.2, qx2 = 0, qx4 = 0, pu1 = 0, pu2 = 0):
    return -((abs(diff_x1)/k)/(1+ (abs(diff_x1)/k)) + (abs(diff_x3)/k)/(1+ (abs(diff_x3)/k)) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2))

def quad_end_pen_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0, sx1 = 10, sx3 = 10, end_horizon = 0):
    return -(qx1*(diff_x1**2) + qx3*(diff_x3**2) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2) + end_horizon*(sx1*(diff_x1**2) + sx3*(diff_x3**2)))

def quad_pen_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0, bound = 0.1, penalty = 50):
    return -(qx1*(diff_x1**2) + qx3*(diff_x3**2) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2) + penalty*(abs(diff_x1) >= bound or abs(diff_x3) >= bound))

def quad_bon_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0, bound = 0.1, bonus = 5):
    return -(qx1*(diff_x1**2) + qx3*(diff_x3**2) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2) - bonus*(abs(diff_x1) <= bound or abs(diff_x3) <= bound))

def angle_normalize_2pi(x):
        return (((x+2*np.pi) % (4*np.pi)) - 2*np.pi) # To keep the angles between -2pi and 2pi

def angle_normalize_pi(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi) # To keep the angles between -pi and pi
