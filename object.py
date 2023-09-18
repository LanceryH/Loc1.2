from dataclasses import dataclass
import numpy as np

@dataclass
class Object:
    position: list[int]
    vitesse: list[int]
    mass: int
    
    def stack_pos_vit(self):
        return np.hstack((self.position,self.vitesse))