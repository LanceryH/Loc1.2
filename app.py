import numpy as np
from tqdm import tqdm
from object import Object
from system import System
from view import View
from dataclasses import dataclass

body_1 = Object(position=[10, 10, 10],
                vitesse=[10, 0, 0],
                mass=100)
body_2 = Object(position=[0, 10, 0],
                vitesse=[10, 0, 0],
                mass=100)
body_3 = Object(position=[10, 0, 0],
                vitesse=[0, 0, 0],
                mass=100)
body_4 = Object(position=[0, 0, 0],
                vitesse=[-10, 10, 0],
                mass=100)
system_solaire = System(bodys=[body_1,body_2,body_4])

Y_RK45, TIME = system_solaire.Resolution_RK45_2(t0=0,tf=2,dt0=1e-5)
Y_RK4 = system_solaire.Resolution_RK4(t0=0,tf=2,N=1000)
Y_RK2 = system_solaire.Resolution_RK2(t0=0,tf=2,N=1000)

color = ["r","g","b","y"]
print(np.shape(Y_RK45))

View(Y=Y_RK45, system_solaire=system_solaire).draw()
