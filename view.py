from dataclasses import dataclass
from flask import Flask
import plotly.graph_objs as go

@dataclass
class View:
    Y: list[list[int]]
    system_solaire: type

    def draw(self):
        app = Flask(__name__, template_folder='templates', static_folder='static')

        fig = go.Figure()

        for index in range(0, len(self.system_solaire.bodys) * 3, 3):
            trace = go.Scatter3d(
                x=self.Y[:, index],
                y=self.Y[:, index + 1],
                z=self.Y[:, index + 2],
                mode='lines',
                name=f'Body {index//3}')
            fig.add_trace(trace)

        fig.update_layout(
            scene=dict(xaxis_title='X',
                       yaxis_title='Y',
                       zaxis_title='Z',
                       aspectmode='cube'),
                       title='Object Trajectories')

        fig.show()