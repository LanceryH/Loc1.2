from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class View:
    Y: list[list[int]]
    system_solaire: type

    def draw(self):
        for index in range(0,np.shape(self.Y)[1]//2,3):
            print(index,index+1)
            plt.plot(self.Y[:,index],self.Y[:,index+1])
        plt.show()