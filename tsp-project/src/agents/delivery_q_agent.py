import numpy as np
import math
from scipy.spatial.distance import cdist

class DeliveryQAgent:
    def __init__(self, xy, max_box, method, boundary_index, boundary_points, mydict, labels, n_cluster, simplices, vertices, states_size, actions_size, alpha=1, beta=1, gamma=0.9, lr=0.6):
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
        self.Q, self.visits, self.Qupdate, self.rewards = self.build_model(states_size, actions_size)
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

    def act(self, s):
        q = np.copy(self.Q[s, :])
        self.timestep += 1
        q[self.states_memory] = -np.inf
        self.states_track = self.states_memory.copy()
        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice([i for i in range(self.actions_size) if i not in self.states_memory])
        self.route += "=>" + str(a)
        self.visits.update({(s, a): self.visits[(s, a)] + 1})
        return a

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []
        self.route = ""

    def updateQ(self):
        self.i += 1
        self.tour_len = 0
        self.lis = []