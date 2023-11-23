from object import Object
from system import System
from view import View
import random
import numpy as np
nb_object=3

body_1 = Object(position=[10, 10, 10],
                vitesse=[10, 0, 0],
                mass=100)
body_2 = Object(position=[0, 10, 0],
                vitesse=[10, 0, 0],
                mass=100)
body_3 = Object(position=[0, 0, 0],
                vitesse=[-10, 10, 0],
                mass=100)

bodys_list = []

for index_i in range(nb_object):
    list_pos_vit = []
    for index_j in range(2):
        randomlist = []
        for i in range(0,3):
            n = random.randint(0,10)
            randomlist.append(n)
        list_pos_vit.append(randomlist)
    bodys_list.append(Object(position=list_pos_vit[0],
                    vitesse=list_pos_vit[1],
                    mass=100))
    
system_solaire = System(bodys=bodys_list)

Y_RK45, TIME = system_solaire.Resolution_RK45_2(t0=0,tf=2,dt0=1e-6)

#Y_RK4 = system_solaire.Resolution_RK4(t0=0,tf=0.5,N=1000)
#Y_RK2 = system_solaire.Resolution_RK2(t0=0,tf=2,N=1000)

View(Y=Y_RK45, system_solaire=system_solaire).draw()

