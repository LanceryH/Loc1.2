import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def force(m1,X1,m2,X2) :
    G=4*np.pi**2    
    d=np.linalg.norm(X1-X2)
    f=-(G*m1*m2/(d**3))*(X1-X2)
    return f
    
class Object:
    def __init__(self, x, y, z, x_d, y_d, z_d, m):
        self.x = x
        self.y = y
        self.z = z
        self.x_d = x_d
        self.y_d = y_d
        self.z_d = z_d
        self.info = [x, y, z, x_d, y_d, z_d]
        self.m = m

class System:
    def __init__(self, Y, M):
        self.Y = np.hstack((Y[0], Y[1], Y[2], Y[3])).reshape((len(M)*6,1))
        self.M = np.array(M)

class Resolution:
    def __init__(self, Y, t0, tf, N, nb_corps, M):
        self.Y = Y
        self.t0 = t0
        self.tf = tf
        self.N = N
        self.nb_corps = nb_corps
        self.M = M
        self.h = (tf-t0)/N
    
    def f(self, t, Y, M):
        nb_corps=len(M)
        F=np.zeros((nb_corps*6,1))
        for i in range(nb_corps) :
            S=np.zeros((3,1))
            for j in range(nb_corps) : 
                if j != i:
                    S+=force(M[i], Y[i*3:(i+1)*3,0].reshape(-1,1),M[j],Y[j*3:(j+1)*3,0].reshape(-1,1))
                
            F[i*3:(i+1)*3,0]=Y[nb_corps*3+i*3:(i+1)*3+nb_corps*3,0]  
            F[nb_corps*3+i*3:(i+1)*3+nb_corps*3,0]=(1/M[i]*S).reshape(-1)    
        return F   

    
    def RK2(self):
        t = self.t0
        Y_return = np.zeros((3*self.nb_corps, self.N))
        for i in tqdm(range(0, self.N)):
            k1 = self.h * self.f(t, self.Y, self.M)
            k2 = self.h * self.f(t + self.h, self.Y + k1, self.M)
            self.Y = self.Y + (k1 + k2) / 2
            t += self.h
            for j in range(0, self.nb_corps):
                Y_return[j*3:(j+1)*3, i] = self.Y[j*3:(j+1)*3, 0]
        return Y_return
    
    def RK4(self):
        t = self.t0
        Y_return = np.zeros((3*self.nb_corps, self.N))
        for i in tqdm(range(0,self.N)) : 
            k1=self.h*self.f(t,self.Y,self.M)
            k2=self.h*self.f(t+self.h/2, self.Y+k1/2,self.M)
            k3=self.h*self.f(t+self.h/2, self.Y+k2/2,self.M)
            k4=self.h*self.f(t+self.h, self.Y+k3,self.M)
            self.Y=self.Y+(1/6)*(k1 +2*k2 +2*k3 + k4)
            t+=self.h    
            for j in range(0,self.nb_corps) :
                Y_return[j*3:(j+1)*3,i]=self.Y[j*3:(j+1)*3,0]
        return Y_return  

body_1 = Object(255,181,0,273,227,0,1000)
body_2 = Object(186, 213, 0, 220, 213, 0, 1000)
body_3 = Object(-16, 23, 0, -20, -1, 0, 1000)
body_4 = Object(3, 32, 0, 1, 18, 0, 1000)

system_solaire = System([body_1.info, body_2.info, body_3.info, body_4.info],
                        [body_1.m, body_2.m, body_3.m, body_4.m])

Y_matrix = Resolution(Y=system_solaire.Y,
                      t0=0,
                      tf=2,
                      N=5000,
                      nb_corps=4,
                      M=system_solaire.M).RK4()

plt.plot(Y_matrix[9,:], Y_matrix[10,:],"b")
plt.plot(Y_matrix[6,:], Y_matrix[7,:],"r")
plt.plot(Y_matrix[3,:], Y_matrix[4,:],"y")
plt.plot(Y_matrix[0,:], Y_matrix[1,:],"g")
plt.show()