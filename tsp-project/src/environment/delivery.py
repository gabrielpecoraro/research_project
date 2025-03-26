import numpy as np
import random
import math
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

class Delivery:
    def __init__(self, xy, boundary_index, n_stops, max_box, fixed, type="nfixed", **kwargs):
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.xy = xy
        self.max_box = max_box
        self.stops = []
        self.fixed = fixed
        self.boundary_index = boundary_index

        self._generate_stops()
        self._generate_q_values()
        self.render()
        self.reset()

    def _generate_stops(self):
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]

    def _generate_q_values(self, box_size=0.2):
        xy = np.column_stack([self.x, self.y])
        self.q_stops = cdist(xy, xy)

    def render(self, return_img=False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        ax.scatter(self.x, self.y, c="red", s=50)

        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        if len(self.stops) > 1:
            ax.plot(self.x[self.stops], self.y[self.stops], c="blue", linewidth=1, linestyle="--")
            xy = self._get_xy(initial=False)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        plt.xticks([])
        plt.yticks([])

        if return_img:
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()

    def reset(self):
        self.stops = []
        if self.fixed:
            k = self.boundary_index
            first_stop = k[0]
        else:
            first_stop = random.randint(0, self.n_stops - 1)
        self.stops.append(first_stop)
        return first_stop

    def step(self, destination):
        state = self._get_state()
        new_state = destination
        reward = self._get_reward(state, new_state)
        self.stops.append(destination)
        return new_state, reward

    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def _get_reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]
        return base_reward