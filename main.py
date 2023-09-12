import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass

def force(m1,X1,m2,X2) :
    G=4*np.pi**2    
    d=np.linalg.norm(X1-X2)
    f=-(G*m1*m2/(d**3))*(X1-X2)
    return f

@dataclass
class Object:
    position: list[int]
    vitesse: list[int]
    mass: int
    
    def stack_pos_vit(self):
        return np.hstack((self.position,self.vitesse))

@dataclass
class System:
    bodys: list[Object]
    nb_corps: int

    def system_pos_vit(self):
        list_int = []
        for index in range(self.nb_corps):
            list_int.append(self.bodys[index].stack_pos_vit())
        return np.hstack((list_int)).reshape((self.nb_corps*6,1))
    
    def system_mass(self):
        list_int = []
        for index in range(self.nb_corps):
            list_int.append(self.bodys[index].mass)
        return list_int
    
    def f(self, t, Y, M):
        nb_corps=len(self.bodys)
        F=np.zeros((nb_corps*6,1))
        for i in range(nb_corps):
            S=np.zeros((3,1))
            for j in range(nb_corps):
                if j != i:
                    S+=force(self.bodys[i].mass, Y[i*3:(i+1)*3,0].reshape(-1,1),M[j],Y[j*3:(j+1)*3,0].reshape(-1,1))

            F[i*3:(i+1)*3,0]=Y[nb_corps*3+i*3:(i+1)*3+nb_corps*3,0]
            F[nb_corps*3+i*3:(i+1)*3+nb_corps*3,0]=(1/M[i]*S).reshape(-1)
        return F

    
    def Resolution_RK2(self, t0, tf, N):
        t = t0
        h = (tf-t0)/N
        Y = self.system_pos_vit()
        M = self.system_masses()
        Y_return = np.zeros((3*self.nb_corps, N))
        for i in tqdm(range(0, N)):
            k1 = h * self.f(t, Y, M)
            k2 = h * self.f(t + h, Y + k1, M)
            Y = Y + (k1 + k2) / 2
            t += h
            for j in range(0, self.nb_corps):
                Y_return[j*3:(j+1)*3, i] = Y[j*3:(j+1)*3, 0]
        return Y_return
    
    def Resolution_RK4(self, t0, tf, N):
        t = t0
        h = (tf-t0)/N
        Y = self.system_pos_vit()
        M = self.system_masses()
        Y_return = np.zeros((3*self.nb_corps, N))
        for i in tqdm(range(0,N)) : 
            k1 = h *self.f(t,Y,M)
            k2 = h *self.f(t+h/2, Y+k1/2,M)
            k3 = h *self.f(t+h/2, Y+k2/2,M)
            k4 = h *self.f(t+h, Y+k3,M)
            Y = Y+(1/6)*(k1+2*k2+2*k3+k4)
            t += h    
            for j in range(0,self.nb_corps) :
                Y_return[j*3:(j+1)*3,i]=Y[j*3:(j+1)*3,0]
        return Y_return 

body_1 = Object(position=[255, 181, 0],
                vitesse=[273, 227, 0],
                mass=1000)
body_2 = Object(position=[186, 213, 0],
                vitesse=[220, 213, 0],
                mass=1000)
body_3 = Object(position=[-16, 23, 0],
                vitesse=[-20, -1, 0],
                mass=1000)
body_4 = Object(position=[3, 32, 0],
                vitesse=[1, 18, 0],
                mass=1000)

system_solaire = System(bodys=[body_1,body_2,body_3,body_4],
                        nb_corps=4)

Y = system_solaire.Resolution_RK4(t0=0,tf=2,N=5000)

color = ["r","g","b","y"]

for index in range(0,system_solaire.nb_corps*3,3):
    plt.plot(Y[index,:], Y[index+1,:],color[index//3])
plt.show()