from numba import jit
import numpy as np
import math

@jit(nopython=True)
def pqr_to_ang_vel(p, q, r, theta, phi):

    phi_dot = (np.cos(phi) / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*p\
                + 0\
                + (np.sin(theta) / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*r          # phi_dot

    theta_dot = (np.sin(phi)*np.sin(theta) / (np.cos(phi)*np.cos(theta)**2 + np.cos(theta)*np.sin(theta)**2))*p\
                + 1*q\
                - (np.sin(phi) / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*r            # theta_dot

    psi_dot = (-np.sin(theta) / (np.cos(phi)*np.cos(theta)**2 + np.cos(theta)*np.sin(theta)**2))*p\
                + 0\
                + (1 / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*r                      # psi_dot

    return phi_dot, theta_dot, psi_dot


@jit(nopython=True)
def get_sderivative(F, M1, M2, M3, S, m, I_xx, I_yy, I_zz, g):

    phi     = S[3]
    theta   = S[4]
    psi     = S[5]
    p       = S[9]
    q       = S[10]
    r       = S[11]
    s_dot   = np.zeros(12)
    
    s_dot[:3] = S[6:9]                                                                    # x_dot, y_dot, z_dot

    phi_dot, theta_dot, psi_dot = pqr_to_ang_vel(p, q, r, theta, phi)
    s_dot[3] = phi_dot

    s_dot[4] = theta_dot

    s_dot[5] = psi_dot

    s_dot[6] = (np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi))*F/m          # x_dot_dot

    s_dot[7] = (np.sin(psi)*np.sin(theta) - np.cos(theta)*np.sin(phi)*np.cos(psi))*F/m          # y_dot_dot

    s_dot[8] = -g + (np.cos(phi)*np.cos(theta))*F/m                                             # z_dot_dot

    s_dot[9] = (1/I_xx) * (M1 - (-I_yy*q*r + I_zz*r*q))                                       # p_dot

    s_dot[10] = (1/I_yy) * (M2 - (I_xx*p*r - I_zz*r*p))                                       # q_dot

    s_dot[11] = (1/I_zz) * (M3 - (-I_xx*p*q + I_yy*q*p))                                      # r_dot

    return s_dot



@jit(nopython = True)
def update(F, M1, M2, M3, curr, t, delta_t, m, I_xx, I_yy, I_zz, g):

    t += delta_t

    k1 = delta_t*get_sderivative(F, M1, M2, M3, curr, m, I_xx, I_yy, I_zz, g)

    k2 = delta_t*get_sderivative(F, M1, M2, M3, curr + k1/4, m, I_xx, I_yy, I_zz, g)

    k3 = delta_t*get_sderivative(F, M1, M2, M3, curr + (3/32)*k1 + (9/32)*k2, m, I_xx, I_yy, I_zz, g)

    k4 = delta_t*get_sderivative(F, M1, M2, M3, curr + (1932/2197)*k1 + (-7200/2197)*k2 + (7296/2197)*k3, m, I_xx, I_yy, I_zz, g)

    k5 = delta_t*get_sderivative(F, M1, M2, M3, curr + (439/216)*k1 + (-8)*k2 + (3680/513)*k3 + (-845/4104)*k4, m, I_xx, I_yy, I_zz, g)

    # curr += (16/135)*k1 + 0*k2 + (6656/12825)*k3 + (28561/56430)*k4 + (-9/50)*k5
    curr += (25/216)*k1 + 0*k2 + (1408/2565)*k3 + (2197/4104)*k4 + (-1/5)*k5

    return curr, t





'''
                (3)            (1)                      x
                     \       /                          |   
                        (O)                             |           
                     /       \                 y________|
                (4)            (2)

                Propeller 1 and 4 are rotating in anticlockwise direction
                Propeller 3 and 2 are rotating in clockwise direction
'''
'''
curr = [x, y, z, phi, theta, psi,
        x_dot, y_dot, z_dot, p, q, r]
'''

class EnvPhysicsEngine:

    def __init__(self):
        self.t = 0.0            # s
        self.m = 1.5            # drone 
        self.I_xx = 0.01        # kg m^2
        self.I_yy = 0.01        # kg m^2
        self.I_zz = 0.01        # kg m^2
        self.g = 9.81           # m s^-2
        self.l = 0.15           # m
        self.f_MF = 0.25        # if battery voltage > 11.5 
        self.curr = np.zeros(12)
        self.delta_t = 0.01     # time step in s

        # useful notations
        self.J = [self.I_xx, 0, 0,
                    0, self.I_yy, 0,
                    0, 0, self.I_zz]
        self.J = np.reshape(self.J, (3,3))

             

    def get_currentState(self):
        return self.curr


    def get_time(self):
        return self.t


    def get_derivative(self, F, M1, M2, M3):

        
        s_dot = get_sderivative(F, M1, M2, M3, self.curr, self.m,
                                self.I_xx, self.I_yy, self.I_zz, self.g)

        return s_dot


    def stepSimulation(self, f1, f2, f3, f4):

        # f1, f2, f3, f4 - Thrust force by each rotor
        F = f1 + f2 + f3 + f4                       # Force in z direction
        M1 = ((f1 + f3) - (f2 + f4))*self.l         # Moment about x-axis (phi)
        M2 = ((f3 + f4) - (f1 + f2))*self.l         # Moment about y-axis (theta)
        M3 = ((f2 + f3) - (f1 + f4))*self.f_MF      # Moment about z-axis (yaw)


       
        # print('forces:-------',F, M1, M2, M3)
        self.curr, self.t = update(F, M1, M2, M3, self.curr, self.t, self.delta_t,
                                   self.m, self.I_xx, self.I_yy, self.I_zz, self.g)
        pass


    
    def veemap(self, R):

        R = np.reshape(R, (9,))
        c = -R[1]
        b = R[2]
        a = -R[5]

        return np.array([a, b, c])


    def rotationalmat(self, phi, theta, psi):

        # arguments are in radian

        R = [np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta), - np.cos(phi) * np.sin(psi) , np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
            np.sin(psi)*np.cos(theta) + np.cos(psi) * np.sin(phi) * np.sin(theta), np.cos(phi)* np.cos(psi), np.sin(psi) * np.sin(theta) - np.cos(theta) * np.sin(phi) * np.cos(psi), 
            -np.sin(theta) * np.cos(phi) , np.sin(phi) , np.cos(phi) * np.cos(theta) ] 
 
        return R


    def isRotationMatrix(self, R) :

        Rt = np.transpose(R)

        shouldBeIdentity = np.dot(Rt, R)

        I = np.identity(3, dtype = R.dtype)

        n = np.linalg.norm(I - shouldBeIdentity)

        return n < 1e-6


    def rpy_from_rotmat(self, R) :

        R = np.reshape(R, (3,3))
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])


    def reset(self, x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot):

        self.t = 0.0
        self.curr[0] = x 
        self.curr[1] = y 
        self.curr[2] = z 
        self.curr[3] = phi 
        self.curr[4] = theta
        self.curr[5] = psi
        self.curr[6] = x_dot
        self.curr[7] = y_dot
        self.curr[8] = z_dot
        p,q,r = self.ang_vel_to_pqr(phi_dot, theta_dot, psi_dot, phi, theta)
        self.curr[9] = p
        self.curr[10] = q
        self.curr[11] = r

    
    def ang_vel_to_pqr(self, phi_dot, theta_dot, psi_dot, phi, theta):
        
        p = np.cos(theta)*phi_dot - np.sin(theta)*np.cos(theta)*psi_dot
        q = theta_dot + np.sin(phi)*psi_dot
        r = np.sin(theta)*phi_dot + np.cos(theta)*np.cos(phi)*psi_dot
        
        return p, q, r


    def pqr_to_ang_vel(self, p, q, r, theta, phi):


        phi_dot = np.cos(theta)*p + np.sin(theta)*r
        theta_dot  = np.tan(phi)*np.sin(theta) * p + q  - np.cos(theta) * np.tan(phi) * r 
        psi_dot = -(np.sin(theta)/ np.cos(phi)) * p  + (np.cos(theta) /np.cos(phi) ) * r

        return phi_dot, theta_dot, psi_dot

    
