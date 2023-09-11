import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')

class Object:
    def __init__(self, x, y, z, x_d, y_d, z_d, m):
        self.x = x
        self.y = y
        self.z = z
        self.x_d = x_d
        self.y_d = y_d
        self.z_d = z_d
        self.inf = [x,y,z,x_d,y_d,z_d]
        self.m = m

class System:
    def __init__(self, Y, M):
        self.Y = np.hstack((Y[0],Y[1],Y[2],Y[3]))
        self.M = np.array(M)

def f(t, Y) :
    mS=330000
    mT=1
    mM=0.1
    mJ=317.83   #0.0123  masse lune
    dST=np.linalg.norm(u[0:3]-u[6:9])
    dSM=np.linalg.norm(u[0:3]-u[12:15])
    dTM=np.linalg.norm(u[6:9]-u[12:15])
    dSJ=np.linalg.norm(u[0:3]-u[18:21])
    dTJ=np.linalg.norm(u[6:9]-u[18:21])
    dMJ=np.linalg.norm(u[12:15]-u[18:21])

    S1= -(4*np.pi**2)*(((mT/mS)/(dST**3))*(u[0]-u[6])+((mM/mS)/(dSM**3))*(u[0]-u[12])+((mJ/mS)/(dSJ**3))*(u[0]-u[18]))
    S2=-(4*np.pi**2)*(((mT/mS)/(dST**3))*(u[1]-u[7])+((mM/mS)/(dSM**3))*(u[1]-u[13])+((mJ/mS)/(dSJ**3))*(u[1]-u[19]))
    S3=-(4*np.pi**2)*(((mT/mS)/(dST**3))*(u[2]-u[8])+((mM/mS)/(dSM**3))*(u[2]-u[14])+((mJ/mS)/(dSJ**3))*(u[2]-u[20]))
    
    S4= -(4*np.pi**2)*((1/(dST**3))*(u[6]-u[0])+((mM/mS)/(dTM**3))*(u[6]-u[12])+((mJ/mS)/(dTJ**3))*(u[6]-u[18]))
    S5= -(4*np.pi**2)*((1/(dST**3))*(u[7]-u[1])+((mM/mS)/(dTM**3))*(u[7]-u[13])+((mJ/mS)/(dTJ**3))*(u[7]-u[19]))
    S6= -(4*np.pi**2)*((1/(dST**3))*(u[8]-u[2])+((mM/mS)/(dTM**3))*(u[8]-u[14])+((mJ/mS)/(dTJ**3))*(u[8]-u[20]))
    
    S7= -(4*np.pi**2)*((1/(dSM**3))*(u[12]-u[0])+((mT/mS)/(dTM**3))*(u[12]-u[6])+((mJ/mS)/(dMJ**3))*(u[12]-u[18]))
    S8= -(4*np.pi**2)*((1/(dSM**3))*(u[13]-u[1])+((mT/mS)/(dTM**3))*(u[13]-u[7])+((mJ/mS)/(dMJ**3))*(u[13]-u[19]))
    S9= -(4*np.pi**2)*((1/(dSM**3))*(u[14]-u[2])+((mT/mS)/(dTM**3))*(u[14]-u[8])+((mJ/mS)/(dMJ**3))*(u[14]-u[20]))
    
    S10= -(4*np.pi**2)*((1/(dSJ**3))*(u[18]-u[0])+((mT/mS)/(dTJ**3))*(u[18]-u[6])+((mM/mS)/(dMJ**3))*(u[18]-u[12]))
    S11= -(4*np.pi**2)*((1/(dSJ**3))*(u[19]-u[1])+((mT/mS)/(dTJ**3))*(u[19]-u[7])+((mM/mS)/(dMJ**3))*(u[19]-u[13]))
    S12= -(4*np.pi**2)*((1/(dSJ**3))*(u[20]-u[2])+((mT/mS)/(dTJ**3))*(u[20]-u[8])+((mM/mS)/(dMJ**3))*(u[20]-u[14]))
    
    return np.array([u[3],u[4],u[5],S1,S2,S3,u[9],u[10],u[11],S4,S5,S6,u[15],u[16],u[17],S7,S8,S9,u[21],u[22],u[23],S10,S11,S12])

t0=0
tf=20
N=800
h=(tf-t0)/float(N)
t=t0

body_1 = Object(0,0,0,0,2,2,330000)
body_2 = Object(1.0167103,0,0,0,6.128,0,1)
body_3 = Object(1.66599116,0,0,0,4.5969,0,0.1)
body_4 = Object(5.46,0,0,0,2.6025,0,317.83)

system_solaire = System([body_1.inf,body_2.inf,body_3.inf,body_4.inf],
                        [body_1.m,body_2.m,body_3.m,body_4.m])

u0=system_solaire.Y
u=u0

for i in range(0,N) : 
    t=t0+h
    k1=h*f(t,u)
    k2=h*f(t+h/2, u+k1/2)
    k3=h*f(t+h/2, u+k2/2)
    k4=h*f(t+h, u+k3)
    u=u+(1/6)*(k1 +2*k2 +2*k3 + k4)
    ax.set_xlim3d(-7,7)
    ax.set_ylim3d(-7,7)
    ax.set_zlim3d(-7,7)
    ax.scatter3D(u[0], u[1], u[2],  c='y')
    ax.scatter3D(u[6], u[7], u[8], c='b')
    ax.scatter3D(u[12], u[13], u[14], c='r')
    ax.scatter3D(u[18], u[19], u[20], c='g')
    plt.pause(0.001)
    plt.cla()
    
plt.show()   

