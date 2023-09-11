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
        self.nb_corps = len(self.M)

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

    
    def Resolution_RK2(self, t0, tf, N):
        t = t0
        h = (tf-t0)/N
        Y_return = np.zeros((3*self.nb_corps, N))
        for i in tqdm(range(0, N)):
            k1 = h * self.f(t, self.Y, self.M)
            k2 = h * self.f(t + h, self.Y + k1, self.M)
            self.Y = self.Y + (k1 + k2) / 2
            t += h
            for j in range(0, self.nb_corps):
                Y_return[j*3:(j+1)*3, i] = self.Y[j*3:(j+1)*3, 0]
        self.Y = Y_return
    
    def Resolution_RK4(self, t0, tf, N):
        t = t0
        h = (tf-t0)/N
        Y_return = np.zeros((3*self.nb_corps, N))
        for i in tqdm(range(0,N)) : 
            k1=h*self.f(t,self.Y,self.M)
            k2=h*self.f(t+h/2, self.Y+k1/2,self.M)
            k3=h*self.f(t+h/2, self.Y+k2/2,self.M)
            k4=h*self.f(t+h, self.Y+k3,self.M)
            self.Y=self.Y+(1/6)*(k1 +2*k2 +2*k3 + k4)
            t+=h    
            for j in range(0,self.nb_corps) :
                Y_return[j*3:(j+1)*3,i]=self.Y[j*3:(j+1)*3,0]
        self.Y = Y_return  


body_1 = Object(255,181,0,273,227,0,1000)
body_2 = Object(186, 213, 0, 220, 213, 0, 1000)
body_3 = Object(-16, 23, 0, -20, -1, 0, 1000)
body_4 = Object(3, 32, 0, 1, 18, 0, 1000)

system_solaire = System(Y=[body_1.info, body_2.info, body_3.info, body_4.info],
                        M=[body_1.m, body_2.m, body_3.m, body_4.m])

system_solaire.Resolution_RK4(t0=0,tf=2,N=5000)

color = ["r","g","b","y"]

for index in range(0,len(system_solaire.M)*3,3):
    plt.plot(system_solaire.Y[index,:], system_solaire.Y[index+1,:],color[index//3])
plt.show()