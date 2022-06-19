import numpy as np
import random
import gym
from gym import spaces
import pybullet as p
from PhysicsEngine import EnvPhysicsEngine, pqr_to_ang_vel




'''

                                     ______________________
        x,y,z               ------> |                      |  ------> f1
            R               ------> |                      |  ------> f2
x_dot,y_dot,z_dot           ------> |      NN Model        |  ------> f3 
phi_dot,theta_dot,psi_dot   ------> |                      |  ------> f4
                                    |______________________|

states (18 total)
f1, f2, f3, f4 are the thrust for by each rotor respectively
'''



class Drone3DEnv(gym.Env):

    metadata = {'render.modes':['human']}

    def __init__(self, drone, marker,
                x_des =0, y_des= 0, z_des=5):

        super(Drone3DEnv, self).__init__()

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(18,), 
                                            dtype=np.float32)

        self.action_space = spaces.Box(low = -1, high = 1,
                                       shape = (4,), 
                                       dtype=np.float32)

        
        self.drone = drone
        self.marker = marker

        self.pos_high = 5                       # m
        self.lin_vel_high = 5                   # m/s
        self.ang_high = np.deg2rad(75)          # It is allowed to attain upto 75 degree, but not more    
        self.ang_vel_high = np.deg2rad(300)     # rad/sec

        self.x_des          = x_des
        self.y_des          = y_des
        self.z_des          = z_des
        self.phi_des        = 0
        self.theta_des      = 0
        self.psi_des        = 0
        self.x_dot_des      = 0
        self.y_dot_des      = 0
        self.z_dot_des      = 0
        self.phi_dot_des    = 0          
        self.theta_dot_des  = 0          
        self.psi_dot_des    = 0          
        

        self.action_high = 1.5*9.81/2  
        # maximum thrust force by each rotor is mg
        self.pe = EnvPhysicsEngine()


    # current state--> current position, orientation, linear and angular velocities
    def state(self):

        x, y, z = self.pe.get_currentState()[0:3]

        phi, theta, psi = self.pe.get_currentState()[3:6]
        

        x_dot, y_dot, z_dot = self.pe.get_currentState()[6:9]

        p, q, r = self.pe.get_currentState()[9:]
        phi_dot, theta_dot, psi_dot = pqr_to_ang_vel(p, q, r, theta, phi)

        # error representation of state
        state = self.abs_to_error_state(x, y, z, phi, theta, psi, \
                                        x_dot, y_dot, z_dot, \
                                        phi_dot, theta_dot, psi_dot)
        return state
    


    # reward
    def reward(self):

        obs = self.state()
        e_x, e_y, e_z = obs[:3]
        phi, theta, psi = self.pe.rpy_from_rotmat(obs[3:12])
                
        e_x_dot, e_y_dot, e_z_dot = obs[12:15]
        e_phi_dot, e_theta_dot, e_psi_dot = obs[15:18]

        pos_err = (e_x**2 + e_y**2 + e_z**2)**0.5
        orn_err = (phi**2 + theta**2 + psi**2)**0.5
        lin_vel_err = (e_x_dot**2 + e_y_dot**2 + e_z_dot**2)**0.5
        ang_vel_err = (e_phi_dot**2 + e_theta_dot**2 + e_psi_dot**2)**0.5
        
        reward = -(10*pos_err + 1*orn_err + 0.5*lin_vel_err + 0.5*ang_vel_err )

        if self.done():
            if (abs(e_x)>=1.5 or abs(e_y)>=1.5 or abs(e_z)>=1.5 or\
                  abs(phi)>=1 or abs(theta)>=1 or abs(psi)>=1 ):

                print('******outbound condition***** e_x {:.2f} e_y {:.2f} e_z {:.2f} \
                        e_phi {:.2f} e_theta {:.2f} e_psi {:.2f}'
                 .format(e_x, e_y, e_z, phi, theta, psi))
                
            else:
                reward = 200.0
                print('*******Desired Condition Achieved*********')

        return float(reward/40.0)
        
    
    # whether goal is achieved or not.
    def done(self):
        obs = self.state()
        e_x, e_y, e_z = obs[:3]
        phi, theta, psi = self.pe.rpy_from_rotmat(obs[3:12])

        e_x_dot, e_y_dot, e_z_dot = obs[12:15]
        e_phi_dot, e_theta_dot, e_psi_dot = obs[15:18]

        if (abs(e_x)>=1.5 or abs(e_y)>=1.5 or abs(e_z)>=1.5 or\
                  abs(phi)>=1 or abs(theta)>=1 or abs(psi)>=1 ):  
            # outbound condition; reset the environment          
            return True

        elif (abs(e_x)          <=0.01/self.pos_high and\
              abs(e_y)          <=0.01/self.pos_high and\
              abs(e_z)          <=0.01/self.pos_high and\
              abs(phi)          <=0.01/self.ang_high and\
              abs(theta)        <=0.01/self.ang_high and\
              abs(psi)          <=0.01/self.ang_high and\
              abs(e_x_dot)      <=0.01/self.lin_vel_high and\
              abs(e_y_dot)      <=0.01/self.lin_vel_high and\
              abs(e_z_dot)      <=0.01/self.lin_vel_high and\
              abs(e_phi_dot)    <=0.01/self.ang_vel_high and\
              abs(e_theta_dot)  <=0.01/self.ang_vel_high and\
              abs(e_psi_dot)    <=0.01/self.ang_vel_high  ):
            # desired condition is achieved
            return True

        return False


    # step
    def step(self, action):

        action_ = (action+1)*self.action_high

        '''for visualization purpose'''
        x, y, z, phi, theta, psi = self.pe.curr[:6]
        p.resetBasePositionAndOrientation(self.drone, [x, y, z],
                            p.getQuaternionFromEuler([phi, theta, psi]))
        p.resetBasePositionAndOrientation(self.marker, 
                            [self.x_des,self.y_des,self.z_des],
                            p.getQuaternionFromEuler([0,0,0]))
        p.stepSimulation()


        '''execution'''
        self.pe.stepSimulation(action_[0], action_[1], action_[2], action_[3])

        state = self.state()
        reward = self.reward()
        done = self.done()
        info = self.info()

        return state, reward, done, info


    # info
    def info(self):
        return {}


    # reset the environment
    def reset(self, obser=None):
        # initializing quadcopter at random angle phi with angular vel phi_dot
        x, y, z, phi, theta, psi,\
            x_dot, y_dot, z_dot, \
              phi_dot, theta_dot, psi_dot =  self.random_state_generator(obser)

        p.resetBasePositionAndOrientation(self.drone, [x,y,z],
                                          p.getQuaternionFromEuler([phi, theta, psi]))

        self.pe.reset(x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot)

        # return state
        state  = self.abs_to_error_state(x, y, z, phi, theta, psi, x_dot, y_dot,
                                         z_dot, phi_dot, theta_dot, psi_dot)
        
        return state


    def random_state_generator(self, obser=None):

        if obser is None:
            
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z = random.uniform(0, 10)

            phi     = random.uniform(-np.pi/3, np.pi/3)
            theta   = random.uniform(-np.pi/3, np.pi/3)
            psi     = random.uniform(-np.pi/3, np.pi/3)

            x_dot  = random.uniform(-1, 1)
            y_dot  = random.uniform(-1, 1)
            z_dot  = random.uniform(-1, 1)

            phi_dot   = random.uniform(-np.pi/6, np.pi/6)
            theta_dot = random.uniform(-np.pi/6, np.pi/6)
            psi_dot   = random.uniform(-np.pi/6, np.pi/6)
             
        else:
            x,y,z, phi,theta,psi,\
              x_dot,y_dot,z_dot,\
                 phi_dot,theta_dot,psi_dot = obser

        return x, y, z, phi, theta, psi,\
                 x_dot, y_dot, z_dot, \
                   phi_dot, theta_dot, psi_dot


    def abs_to_error_state(self, x, y, z, phi, theta,
                             psi, x_dot, y_dot, z_dot,
                             phi_dot, theta_dot, psi_dot):

        # assuming maximum quadcopter angle would be 90 degree
        e_x = (x - self.x_des) / self.pos_high
        e_y = (y - self.y_des) / self.pos_high
        e_z = (z - self.z_des) / self.pos_high

        R = self.pe.rotationalmat(phi, theta, psi)

        e_x_dot = (x_dot - self.x_dot_des) / self.lin_vel_high
        e_y_dot = (y_dot - self.y_dot_des) / self.lin_vel_high
        e_z_dot = (z_dot - self.z_dot_des) / self.lin_vel_high

        e_phi_dot = (phi_dot - self.phi_dot_des) / self.ang_vel_high
        e_theta_dot = (theta_dot - self.theta_dot_des) / self.ang_vel_high
        e_psi_dot = (psi_dot - self.psi_dot_des) / self.ang_vel_high

        return np.array([e_x, e_y, e_z,   \
                        R[0], R[1], R[2], \
                        R[3], R[4], R[5], \
                        R[6], R[7], R[8], \
                        e_x_dot, e_y_dot, e_z_dot,\
                        e_phi_dot, e_theta_dot, e_psi_dot])



