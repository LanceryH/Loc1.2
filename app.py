import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from flask import Flask, request, jsonify, render_template
from flask_table import Table, Col
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

app = Flask(__name__, template_folder='templates', static_folder='static')

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
        M = self.system_mass()
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
        M = self.system_mass()
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
    
    def Resolution_RK45(self, t0, tf, N):
        t = t0
        pas_var = []
        error_min = 1e-3
        h = (tf-t0)/N
        Y = self.system_pos_vit()
        M = self.system_mass()
        Y_return = np.zeros((3*self.nb_corps, N))
        for i in tqdm(range(0,N)) : 
            k1 = h *self.f(t,Y,M)
            k2 = h *self.f(t+h/4, Y+k1/4,M)
            k3 = h *self.f(t+h*3/8, Y+k1*3/32+k2*9/32,M)
            k4 = h *self.f(t+h*12/13, Y+k1*1932/2197-k2*7200/2197+k3*7296/2197,M)
            k5 = h *self.f(t+h, Y+k1*439/216-k2*8+k3*3680/513-k4*845/4104,M)
            k6 = h *self.f(t+h/2, Y-k1*8/27+k2*2-k3*3544/2565+k4*1859/4104-k5*11/40,M)
            Y_RK4 = Y+k1*25/216+k2*0+k3*1408/2565+k4*2197/4101-k5*1/5
            Y_RK5 = Y+k1*16/135+k2*0+k3*6656/12825+k4*28561/56430-k5*9/50+k6*2/55
            t += h
            error = np.linalg.norm(np.abs(Y_RK4-Y_RK5))
            q = (error_min/error)**(1/5)
            pas_var.append(h)
            h = q*h
            Y = Y_RK5
            for j in range(0,self.nb_corps) :
                Y_return[j*3:(j+1)*3,i]=Y[j*3:(j+1)*3,0]
        return Y_return
    
    def Resolution_RK45_2(self, t0, tf, dt0):
        t = t0
        time = []
        error_min = 1e-3
        h = dt0
        Y = self.system_pos_vit()
        M = self.system_mass()
        Y_return = []
        while t < tf : 
            #print(t)
            k1 = h *self.f(t,Y,M)
            k2 = h *self.f(t+h/4, Y+k1/4,M)
            k3 = h *self.f(t+h*3/8, Y+k1*3/32+k2*9/32,M)
            k4 = h *self.f(t+h*12/13, Y+k1*1932/2197-k2*7200/2197+k3*7296/2197,M)
            k5 = h *self.f(t+h, Y+k1*439/216-k2*8+k3*3680/513-k4*845/4104,M)
            k6 = h *self.f(t+h/2, Y-k1*8/27+k2*2-k3*3544/2565+k4*1859/4104-k5*11/40,M)
            Y_RK4 = Y+k1*25/216+k2*0+k3*1408/2565+k4*2197/4101-k5*1/5
            Y_RK5 = Y+k1*16/135+k2*0+k3*6656/12825+k4*28561/56430-k5*9/50+k6*2/55
            t += h
            error = np.linalg.norm(np.abs(Y_RK4-Y_RK5))
            q = (error_min/error)**(1/5)
            time.append(t)
            h = q*h
            Y = Y_RK5
            for j in range(0,self.nb_corps) :
                del j
                Y_return.append(Y[:,0])
        return np.array(Y_return), time

class Results(Table):
    id = Col('Id', show=False)
    artist = Col('Artist')
    title = Col('Title')
    release_date = Col('Release Date')
    publisher = Col('Publisher')
    media_type = Col('Media')

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

Y_RK45, TIME = system_solaire.Resolution_RK45_2(t0=0,tf=2,dt0=1e-3)
#Y_RK4 = system_solaire.Resolution_RK4(t0=0,tf=2,N=1000)
#Y_RK2 = system_solaire.Resolution_RK2(t0=0,tf=2,N=1000)

def update(frame):
    ax.clear()
    ax.set_xlim(-300, 300)  # Adjust the limits based on your system's dimensions
    ax.set_ylim(-300, 300)  # Adjust the limits based on your system's dimensions
    ax.set_aspect('equal')  # Ensure equal aspect ratio for the plot

    # Update the positions of celestial bodies at the given frame
    for i, body in enumerate(system_solaire.bodys):
        ax.scatter(Y_RK45[frame, i * 3], Y_RK45[frame, i * 3 + 1], label=f'Body {i + 1}', s=100)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Time Step: {frame}')

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Create the animation
ani = FuncAnimation(fig, update, frames=len(Y_RK45), interval=100)  # Adjust interval as needed

# To display the animation in a Jupyter Notebook, you can use the following line:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save the animation as a video file (e.g., .mp4), you can use the following line:
# ani.save('celestial_bodies_animation.mp4', writer='ffmpeg')

# To display the animation using a standalone Matplotlib window:
plt.show()