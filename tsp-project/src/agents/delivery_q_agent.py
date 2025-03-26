import numpy as np
import math
from scipy.spatial.distance import cdist


class DeliveryQAgent:
    def __init__(
        self,
        xy,
        max_box,
        method,
        boundary_index,
        boundary_points,
        mydict,
        labels,
        n_cluster,
        simplices,
        vertices,
        states_size,
        actions_size,
        alpha=1,
        beta=1,
        gamma=0.9,
        lr=0.6,
    ):
        self.xy = xy
        self.method = method
        self.boundary_index = boundary_index
        self.boundary_points = boundary_points
        self.labels = labels
        self.mydict = mydict
        self.n_cluster = n_cluster
        self.boundary_memory = []
        self.simplices = simplices
        self.vertices = vertices
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = 1
        self.gamma = gamma
        self.lr = lr
        self.max_box = max_box
        self.iter_num = 1
        self.Q, self.visits, self.Qupdate, self.rewards = self.build_model(
            states_size, actions_size
        )
        self.route = ""
        self.end = 0
        self.timestep = 0
        self.c = 500
        self.tour_len = 0
        self.min_tour = 1000
        self.i = 0
        self.states_track = []
        self.best = ""
        self.lis = []
        self.al = alpha
        self.a = 1
        self.beta = beta
        self.mult = 1
        self.change = 0
        self.num = 0
        self.best_tour = [(-1, 1)]
        self.once = True
        self.Q2 = self.Q.copy()
        self.memory = []

    def build_model(self, states_size, actions_size):
        Q = cdist(self.xy, self.xy)
        visits = {}
        update = {}
        rewards = {}
        for i in range(states_size):
            Q[i, i] = -np.inf
            for j in range(states_size):
                visits[(i, j)] = 0
                update[(i, j)] = 0
                rewards[(i, j)] = 0
                if j == i:
                    Q[i, j] = -np.inf
        return Q, visits, update, rewards

    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.actions_size)
        else:
            # Exploit: choose the best action based on Q-values
            return np.argmax(self.Q[state])

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        """Reset the agent's memory at the start of each episode"""
        self.memory = []
        self.states_memory = []
        self.route = ""

    def train(self, state, new_state, reward):
        """Update Q-values using Q-learning algorithm"""
        if self.method == "cluster":
            # Q-learning update rule
            old_value = self.Q[state, new_state]
            next_max = np.max(self.Q[new_state])
            new_value = (1 - self.lr) * old_value + self.lr * (
                reward + self.gamma * next_max
            )
            self.Q[state, new_state] = new_value

    def updateQ(self):
        self.i += 1
        self.tour_len = 0
        self.lis = []
