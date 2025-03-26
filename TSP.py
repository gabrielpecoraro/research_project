# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 02:12:19 2023

@author: khali
"""

"""
code by Khalid Talal Suliman
Created on Thu Apr 27 02:12:19 2023
Final code for SRP
Project: Solving the traveling salesman problem using machine learning (Reinforcement Learning)
Compare different methods with optimal solution
methods : Nearest - Neighbour / Epsilon greedy / UCB / Clustering RL
"""

# PACKAGES
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import random
from scipy.spatial.distance import cdist
import imageio
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from tqdm import tqdm_notebook
import math
from statistics import mean

nodes = [5, 10, 15, 20, 30]  # number of nodes/stops/city
rep = 10  # number of repeats at each n
max_box = 20  # scale of points
methods = ["cluster", "epsilon", "ucb", "nn"]
# methods = ["ucb"]


# important functions used below


# used to give tour length reward
def actfun(x, n_stops, max_box):
    x = float(x)
    return 5 * (n_stops * max_box / 2 - x)


# calc distance btwn points
def distance(coordinates, city1, city2):
    c1 = coordinates[city1]
    c2 = coordinates[city2]
    diff = (c1[0] - c2[0], c1[1] - c2[1])
    return math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])


def diff(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return (dx, dy)


# ENV


class Delivery(object):
    def __init__(
        self, xy, boundary_index, n_stops, max_box, fixed, type="nfixed", **kwargs
    ):
        # print(f"Initialized Delivery Environment with {len(coordinates)} random stops")

        # Initialization
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.xy = xy
        self.max_box = max_box
        self.stops = []
        self.fixed = fixed
        self.boundary_index = boundary_index

        # Generate stops
        self._generate_stops()
        self._generate_q_values()
        self.render()
        # Initialize first point
        self.reset()

        # create stops (customers)

    def _generate_stops(self):
        # seperate the x and y
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]

    def _generate_q_values(self, box_size=0.2):
        # Generate actual Q Values corresponding to time elapsed between two points
        xy = np.column_stack([self.x, self.y])  # remakes the xy coordinates
        self.q_stops = cdist(xy, xy)  # calc distance between stops

    # show the stop (still not sure!!!)
    def render(self, return_img=False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        ax.scatter(self.x, self.y, c="red", s=50)

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        # Show itinerary
        if len(self.stops) > 1:
            ax.plot(
                self.x[self.stops],
                self.y[self.stops],
                c="blue",
                linewidth=1,
                linestyle="--",
            )

            # Annotate END
            xy = self._get_xy(initial=False)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        if hasattr(self, "box"):
            left, bottom = self.box[0], self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left, bottom), width, height)
            collection = PatchCollection([rect], facecolor="red", alpha=0.2)
            ax.add_collection(collection)

        plt.xticks([])
        plt.yticks([])

        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()

            # resets the env

    def reset(self):
        # Stops placeholder
        self.stops = []

        # Random first stop
        # this is changed first_stop = 1
        # first_stop =  random.randint(0,self.n_stops-1)
        if self.fixed:
            # first_stop = 0
            k = self.boundary_index
            first_stop = k[0]
        else:
            first_stop = random.randint(0, self.n_stops - 1)
        self.stops.append(first_stop)

        return first_stop

    # choose action
    def step(self, destination):
        # Get current state
        state = self._get_state()  # observe current state
        new_state = destination

        # Get reward for such a move
        reward = self._get_reward(state, new_state)

        # Append new_state to stops
        self.stops.append(destination)
        # done = len(self.stops) == self.n_stops +1

        return new_state, reward

    def _get_state(self):
        return self.stops[-1]  # last element in seq

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def actfun1(self, x):
        x = float(x)
        k = math.sqrt((self.max_box**2) * 2)  # max distance btw two points
        return 3 * (k - x)

    def _get_reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]  # let reward = distance
        return base_reward


# Agent


class DeliveryQAgent(object):
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
        # self.epsilon_decay = 1/num_eps
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
        # self.initial_eps = epsilon
        self.once = True
        self.Q2 = self.Q.copy()

    def remember(self, *args):
        self.memory.save(args)

    # plot current tour if error occurs
    def plotcurrenttour(self, xy, list1, n, simplices, vertices, labels):
        x = xy[:, 0]
        y = xy[:, 1]
        coordinates = {}
        for i in range(n):
            coordinates["city" + str(i + 1)] = (x[i], y[i])
        # plt.scatter(x, y, c ="blue")

        count = 1
        # env.stops.pop(len(list1)-1)
        if self.method == "cluster":
            points = np.column_stack([x, y])  # remakes the xy coordinates
            plt.scatter(x, y, c=labels, s=50)

        tour = list1
        for i in tour:
            if i == tour[-1]:
                k = "city" + str(i + 1)
                g = "city" + str(tour[0] + 1)
                plt.arrow(
                    coordinates[k][0],
                    coordinates[k][1],
                    diff(coordinates[k], coordinates[g])[0],
                    diff(coordinates[k], coordinates[g])[1],
                    head_width=0.15,
                )
                plt.text(coordinates[k][0], coordinates[k][1], s=str(count))

            else:
                k = "city" + str(i + 1)
                g = "city" + str(tour[count] + 1)
                plt.text(coordinates[k][0], coordinates[k][1], s=str(count))
                plt.arrow(
                    coordinates[k][0],
                    coordinates[k][1],
                    diff(coordinates[k], coordinates[g])[0],
                    diff(coordinates[k], coordinates[g])[1],
                    head_width=0.15,
                )
                count += 1
        plt.title("current tour")
        plt.show()

    import math

    def chosen_point(self, x, point_list):
        weights = []
        for point in point_list:
            distance = math.sqrt(
                (self.xy[point][0] - self.xy[x][0]) ** 2
                + (self.xy[point][1] - self.xy[x][1]) ** 2
            )  # calculate the Euclidean distance between x and the current point
            weight = 1 / distance  # smaller distance leads to higher weight
            weights.append(weight)
        total_weight = sum(weights)  # calculate the sum of all weights
        probabilities = [
            weight / total_weight for weight in weights
        ]  # normalize the weights to obtain probabilities
        chosen_point = np.random.choice(point_list, p=probabilities)
        # chosen_point = point_list[np.random.choice(len(point_list), p=probabilities)] # choose a point from the list using numpy random choice with the probabilities
        return chosen_point

    def build_model(self, states_size, actions_size):
        # Q = np.zeros([states_size,actions_size])
        if self.method == "cluster":
            x = math.sqrt((self.max_box**2) * 2)
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

                        continue
                    if i in self.boundary_index:
                        if j in self.boundary_index:
                            Q[i, j] = 3 * (x - Q[i, j])
                        elif self.labels[i] == self.labels[j]:
                            Q[i, j] = 3 * (x - Q[i, j])

                        else:
                            Q[i, j] = -np.inf
                    else:
                        if self.labels[i] == self.labels[j]:
                            Q[i, j] = 3 * (x - Q[i, j])
                        else:
                            Q[i, j] = -np.inf

        else:
            x = math.sqrt((self.max_box**2) * 2)

            Q = cdist(self.xy, self.xy)
            visits = {}
            update = {}
            rewards = {}
            for i in range(states_size):
                Q[i, i] = -np.inf
                for j in range(states_size):
                    if j == i:
                        Q[i, j] = -np.inf
                    else:
                        Q[i, j] = 3 * (x - Q[i, j])

                        visits[(i, j)] = 0
                        update[(i, j)] = 0
                        rewards[(i, j)] = 0
        print(Q)

        return Q, visits, update, rewards

    # update the Q-values
    def train(self, s, a, r):
        # Get Q Vector

        self.lis.append((s, a))
        self.rewards.update({(s, a): r})

    def count(self, s):
        lab = self.labels[s]
        g = self.mydict[lab].copy()

        k = [i for i in g if i not in self.states_memory]
        bound = []
        inter = []
        for l in k:
            if l in self.boundary_index:
                bound.append(l)
            else:
                inter.append(l)

        all_bound = [i for i in self.boundary_index if i not in self.states_memory]
        """
        print("########################")    
        print(k)
        print(bound)
        print(inter)
        print(all_bound)
        """
        # k is list of all points left in cluster
        # bound is list of all boundary points left in cluster
        # inter is list of all interior points left in cluster
        # all_bound is list of all boundary points left in env
        return k, bound, inter, all_bound

    def act(self, s):
        # Get Q Vector
        if self.method == "cluster":
            exploit = False
            q = np.copy(self.Q[s, :])
            # print("##############################################")

            # print(q)
            self.timestep += 1
            q[self.states_memory] = -np.inf
            self.states_track = self.states_memory.copy()
            # lab = self.cluster_memory[s]

            # print("Tour position number :"+str(len(self.states_memory))+" Iteration num "+str(self.i) )
            if len(self.states_memory) > self.actions_size:
                k = 0
            # print(s)
            if self.i == self.num - 1:
                a = np.argmax(q)
            else:
                if np.random.rand() > self.epsilon:
                    # print("EPSILON")
                    exploit = True
                    # a = np.argmax(q)

                if self.end == 1:
                    self.boundary_memory.pop(0)
                    cl_list, bound, inter, all_bound = self.count(s)
                    a = all_bound[0]
                else:
                    if s in self.boundary_index:
                        cl_list, bound, inter, all_bound = self.count(s)
                        """
                        print("Boundary point")
                        print("Cluster points",cl_list)
                        print("Cluster bound points", bound)
                        print("Cluster inter points",inter)
                        print("All bound points",all_bound)
                        """
                        if len(bound) == 1 and len(inter) > 0:
                            if exploit:
                                for i in range(self.states_size):
                                    if i in inter:
                                        continue
                                    else:
                                        q[i] = -np.inf

                                # print("Q-values for inter ",q[inter])
                                a = np.argmax(q)
                                # print(self.xy[a])

                            else:
                                # a = np.random.choice([x for x in inter])
                                a = self.chosen_point(s, inter)

                        else:
                            lis = all_bound + cl_list
                            k = set(lis)
                            lis = list(k)
                            lab = self.labels[s]
                            # print("the lab of s is ", lab)
                            k = [
                                i
                                for i in range(self.states_size)
                                if i not in self.states_memory
                            ]
                            l = [self.labels[i] for i in k]
                            # print("the labs of un is ",l)
                            if len(lis) == 0:
                                k = [
                                    i
                                    for i in range(self.states_size)
                                    if i not in self.states_memory
                                ]
                                self.plotcurrenttour(
                                    self.xy,
                                    self.states_memory,
                                    self.actions_size,
                                    self.method,
                                    self.simplices,
                                    self.vertices,
                                    self.labels,
                                )
                                k = 0
                                # print(self.Q_5)
                                # print(self.Q_100)

                            if exploit:
                                # print("Q=values for lis ",q[lis])
                                for i in range(self.states_size):
                                    if i in lis:
                                        continue
                                    else:
                                        q[i] = -np.inf
                                a = np.argmax(q)
                            else:
                                # a = np.random.choice([x for x in lis])
                                a = self.chosen_point(s, lis)

                    else:
                        cl_list, bound, inter, all_bound = self.count(s)
                        """
                        print("Interior point")
                        print("Cluster points",cl_list)
                        print("Cluster bound points", bound)
                        print("Cluster inter points",inter)
                        print("All bound points",all_bound)
                        
                        """
                        if len(bound) == 1:
                            if len(inter) > 0:
                                if exploit:
                                    # print("Q-values for inter ",q[inter])
                                    for i in range(self.states_size):
                                        if i in inter:
                                            continue
                                        else:
                                            q[i] = -np.inf

                                    a = np.argmax(q)
                                else:
                                    # a = np.random.choice([x for x in inter])

                                    a = self.chosen_point(s, inter)

                            else:
                                a = bound[0]
                        else:
                            """
                            print(self.timestep)
                            print(s)
                            print(s in self.boundary_index)
                            print(self.mydict[self.labels[s]])
                            print(cl_list,bound,inter,all_bound)
                            """
                            if len(cl_list) == 0:
                                k = [
                                    i
                                    for i in range(self.states_size)
                                    if i not in self.states_memory
                                ]
                                self.plotcurrenttour(
                                    self.xy,
                                    self.states_memory,
                                    self.actions_size,
                                    self.method,
                                    self.simplices,
                                    self.vertices,
                                    self.labels,
                                )
                                # print(self.Q_5)
                                # print(self.Q_100)
                                k = 0
                            # print("Q-values for cl_list ",q[cl_list])

                            if exploit:
                                # print("Q-values for cl_list ",q[cl_list])
                                for i in range(self.states_size):
                                    if i in cl_list:
                                        continue
                                    else:
                                        q[i] = -np.inf

                                a = np.argmax(q)
                            else:
                                # a = np.random.choice([x for x in inter])

                                a = self.chosen_point(s, cl_list)

                            # a = np.random.choice([x for x in cl_list])
            s1 = self.states_memory.copy()
            s1.sort()
            if a in s1:
                k = 0
            """
            
            if self.end != 1 :    
                lab = self.labels[a]        
                self.cluster_memory[lab].append(a)
                print("end ",self.cluster_memory[lab])
                print("end ", self.cluster_memory)
                print(lab)
                """
        elif self.method == "epsilon":
            q = np.copy(self.Q[s, :])
            self.timestep += 1
            q[self.states_memory] = -np.inf

            # Avoid already visited states
            self.states_track = self.states_memory.copy()
            if self.i == self.num - 1:
                a = np.argmax(q)
            else:
                if np.random.rand() > self.epsilon:
                    a = np.argmax(q)
                else:
                    if self.end == 1:
                        self.states_track.pop(0)
                    # a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_track])
                    lis = [
                        i for i in range(self.states_size) if i not in self.states_track
                    ]

                    a = self.chosen_point(s, lis)

        elif self.method == "ucb":
            self.timestep += 1
            q = np.copy(self.Q[s, :])
            q[self.states_memory] = -np.inf
            self.states_track = self.states_memory.copy()

            if self.i == self.num - 1:
                a = np.argmax(q)
            else:
                if np.random.rand() > self.epsilon:
                    a = np.argmax(q)
                else:
                    if self.end == 1:
                        self.states_track.pop(0)
                    # a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_track])

                    for i in range(self.actions_size):
                        if i in self.states_track:
                            continue
                        q[i] = -q[i] + self.c * math.sqrt(
                            math.log(self.timestep) / (self.visits[(s, i)] + 1)
                        )
                    # print(q)
                    a = np.argmax(q)
        elif self.method == "nn":
            q = np.copy(self.Q2[s, :])
            q[self.states_memory] = -np.inf
            a = np.argmax(q)

        self.route = self.route + "=>" + str(a)
        self.visits.update({(s, a): self.visits[(s, a)] + 1})
        return a

    def remember_state(self, s):
        if self.once:
            print("boundary points are ", self.boundary_index)
            self.once = False

        self.states_memory.append(s)
        if self.method == "cluster":
            lab = self.labels[s]
            # print(lab)

            self.cluster_memory[lab].append(s)
            # print("in rem ",self.cluster_memory[lab])
            type(self.cluster_memory)
            type(self.cluster_memory[lab])

            if s in self.boundary_index:
                self.boundary_memory.append(s)
            # print(self.boundary_memory)
        # self.states_track = self.states_memory

    def reset_memory(self):
        self.states_memory = []
        # self.states_track = []
        if self.method == "cluster":
            self.cluster_memory = {i: list() for i in range(self.n_cluster)}
            self.boundary_memory = []
        self.route = ""

    def updateQ(self):
        if self.i == 5:
            print(self.Q)

        self.visited = set()
        # print(self.states_memory)
        # print(self.epsilon)

        if self.tour_len <= self.min_tour:
            self.min_tour = self.tour_len
            self.best = self.route
            self.best_tour = self.lis
            self.mult = 1

            self.change += 1
            Qchange = {}

            for i in self.lis:
                k = self.Q[i]
                s = str(self.Q[i]) + " => "
                # print(i)
                self.visited.add(i[0])
                self.visited.add(i[1])
                if i == self.lis[-1]:
                    self.Q[i] = self.Q[i] + (self.lr + 0.3) * (
                        (self.beta * self.rewards[i])
                        + actfun(self.tour_len, self.actions_size, self.max_box)
                        * self.al
                        * self.mult
                        - self.Q[i]
                    )
                    continue

                if i == self.lis[-2]:
                    self.visited.remove(self.states_memory[0])

                q = self.Q[
                    i[1], [x for x in range(self.actions_size) if x not in self.visited]
                ]
                # print(np.mean(q))
                # print(q)

                self.Q[i] = self.Q[i] + (self.lr + 0.3) * (
                    (self.beta * self.rewards[i])
                    + actfun(self.tour_len, self.actions_size, self.max_box)
                    * self.al
                    * self.mult
                    + self.gamma * np.max(q)
                    - self.Q[i]
                )
                s = s + str(self.Q[i])
                Qchange[i] = s
                if np.isinf(self.Q[i]):
                    l = 0
                    # print("Cluster : ",self.cluster)
                    print("state ", i)
                    print("old q-value ", k)
                    print("reward ", self.rewards[i])
                    print("reward from tour len ", actfun(self.tour_len))
                    print(self.al)
                    print(self.beta)
                    print(self.mult)
                if np.isnan(self.Q[i]):
                    l = 0
                    # print("Cluster : ",self.cluster)
                    print("state ", i)
                    print("old q-value ", k)
                    print("reward ", self.rewards[i])
                    print("reward from tour len ", actfun(self.tour_len))
                    print(self.al)
                    print(self.beta)
                    print(self.mult)
                """
            for i in Qchange.keys() :
                print(i," : ",Qchange[i])
            print("#########################")
            """
        else:
            """
            Qchange = {}

            for i in self.lis:
                k = self.Q[i]
                s = str(self.Q[i])+" => "
               # print(i)
                self.visited.add(i[0])
                self.visited.add(i[1])
                if i == self.lis[-1]:
                   self.Q[i] = self.Q[i] + (self.lr -0.3)*( (self.beta*self.rewards[i]) +actfun(self.tour_len,self.actions_size,self.max_box)*self.al*self.mult- self.Q[i]) 
                   continue
                        
                if i == self.lis[-2]:
                    self.visited.remove(self.states_memory[0])

                q = self.Q[i[1],[x for x in range(self.actions_size) if x not in self.visited]]
                #print(np.mean(q))
                #print(q)
               
                self.Q[i] = self.Q[i] + (self.lr -0.3)*( (self.beta*self.rewards[i]) + actfun(self.tour_len,self.actions_size,self.max_box)*self.al*self.mult+ self.gamma*np.max(q) - self.Q[i]) 
                s = s+str(self.Q[i])
                Qchange[i] = s
                if np.isinf(self.Q[i]):
                    l = 0
                    #print("Cluster : ",self.cluster)
                    print("state ", i)
                    print("old q-value ",k)
                    print("reward ",self.rewards[i])
                    #print("reward from tour len ",actfun(self.tour_len))
                    print(self.al)
                    print(self.beta)
                    print(self.mult)
                if np.isnan(self.Q[i]):
                    l = 0
                    #print("Cluster : ",self.cluster)
                    print("state ", i)
                    print("old q-value ",k)
                    print("reward ",self.rewards[i])
                   # print("reward from tour len ",actfun(self.tour_len))
                    print(self.al)
                    print(self.beta)
                    print(self.mult)
          
        if np.isnan(self.Q).any():
            # create a Boolean matrix where True values represent the presence of NaN
            nan_mask = np.isnan(self.Q)

            # get the indices of True values using the where() function
            nan_indices = np.where(nan_mask)

            # print the indices where NaN values are present
            l = []
            for i in range(len(nan_indices[0])):
                row_index = nan_indices[0][i]
                col_index = nan_indices[1][i]
                l.append((row_index,col_index))
                #print(f"Index ({row_index}, {col_index}) has NaN value")
            for i in l :
                print("########################")
                print(i)
                print("Old q-value ", self.Q[i])
                print("Reward ", self.rewards[i]) 
        """
        if self.tour_len < self.min_tour:
            self.min_tour = self.tour_len
            self.best = self.route
        self.i += 1
        self.tour_len = 0
        self.lis = []


def run_episode(env, agent, verbose=1):
    s = env.reset()
    agent.reset_memory()
    agent.route = agent.route + str(s)
    episode_reward = 0
    done = False
    while done == False:
        # Remember the states
        agent.remember_state(s)
        # Choose an action
        a = agent.act(s)
        # Take the action, and get the reward from environment
        s_next, r = env.step(a)
        episode_reward += r
        agent.tour_len += r
        # Tweak the reward
        # r = -1 * r
        agent.train(s, a, env.actfun1(r))

        if verbose:
            print(s_next, r, done)
        check1 = True
        for i in range(env.action_space):
            if i in env.stops:
                continue
            else:
                check1 = False
                break
        # len(env.stops) == env.n_stops
        if check1:
            s_next, r = env.step(agent.states_memory[0])
            episode_reward += r
            agent.tour_len += r
            agent.train(a, s_next, env.actfun1(r))
            agent.route = agent.route + "=>" + str(s_next)

            done = True
            # agent.states_track =  agent.states_memory.pop(0)
        # Update our knowledge in the Q-table
        # Update the caches
        # episode_reward += r
        s = s_next
        # print(agent.states_memory)
        # print(len(env.stops)
        # print(i,done)

        # agent.epsilon -= 0.0009
        # If the episode is terminated
        # print(agent.route)
    for i in range(env.action_space):
        if i in env.stops:
            continue
        else:
            print("ERROR")

    k = agent.Q
    agent.updateQ()
    agent.epsilon -= 1 / agent.num
    """
    if agent.fixed == False and agent.epsilon > 0.00055:
            #agent.epsilon -= 0.0011
        agent.epsilon -= 0.00055
        """
    return env, agent, episode_reward


def plottour(x, y, list1, n, r, method, simplices, vertices, labels):
    coordinates = {}
    for i in range(n):
        coordinates["city" + str(i + 1)] = (x[i], y[i])
    # plt.scatter(x, y, c ="blue")
    count = 1
    # env.stops.pop(len(list1)-1)
    X = np.column_stack([x, y])  # remakes the xy coordinates

    if method == "cluster":
        plt.scatter(x, y, c=labels, s=50)
    else:
        plt.scatter(x, y, s=50)
    c = 1
    for i in range(len(set(labels))):
        points = X[labels == i]
        if len(points) < 3:
            continue
        plt.plot(points[simplices[c], 0], points[simplices[c], 1], "c")
        plt.plot(
            points[vertices[c], 0],
            points[vertices[c], 1],
            "o",
            mec="r",
            color="none",
            lw=1,
            markersize=10,
        )
        c += 1

    list1.pop()
    tour = list1
    for i in tour:
        if i == tour[-1]:
            k = "city" + str(i + 1)
            g = "city" + str(tour[0] + 1)
            plt.arrow(
                coordinates[k][0],
                coordinates[k][1],
                diff(coordinates[k], coordinates[g])[0],
                diff(coordinates[k], coordinates[g])[1],
                head_width=0.15,
            )
            plt.text(coordinates[k][0], coordinates[k][1], s=str(count))

        else:
            k = "city" + str(i + 1)
            g = "city" + str(tour[count] + 1)
            plt.text(coordinates[k][0], coordinates[k][1], s=str(count))
            plt.arrow(
                coordinates[k][0],
                coordinates[k][1],
                diff(coordinates[k], coordinates[g])[0],
                diff(coordinates[k], coordinates[g])[1],
                head_width=0.15,
            )
            count += 1

        plt.title("RL-TSP-solver method " + method + " dist = " + str(round(r, 2)))
    plt.show()


def run_n_episodes(
    env, agent, plot1, name="training.gif", n_episodes=1000, render_each=10, fps=10
):
    # Store the rewards
    rewards = []
    imgs = []
    check = True
    # Experience replay
    for i in tqdm_notebook(range(n_episodes)):
        agent.num = n_episodes
        if check:
            if agent.epsilon < 0.1:
                print(i)
                check = False
        if i == n_episodes - 1:
            k = agent.epsilon
            agent.epsilon = 0
        # Run the episode
        env, agent, episode_reward = run_episode(env, agent, verbose=0)
        rewards.append(episode_reward)
        # print("episode "+str(i)+" worked")
        if i % render_each == 0:
            img = env.render(return_img=True)
            imgs.append(img)

    # Show rewards
    plt.figure(figsize=(15, 3))
    # plt.title("Rewards over training")

    # plt.title("Tour distance  method "+agent.method+" best dist = "+str(round(agent.min_tour,2)))

    plt.plot(rewards)
    plt.show()
    """
    if agent.ucb:
        print(agent.visits)
        print(agent.Q)
    print(agent.mult)
    print(agent.change)
    print("Last tour ",agent.route)
    print("Best tour ",agent.best)
    print(agent.min_tour)
    print(agent.Q)
    
    # Save imgs as gif
    imageio.mimsave(name,imgs,fps = fps)
    print("The distance of the tour using RL solver is: "+str(round(-episode_reward,2)))
    """
    if plot1:
        plottour(
            env.x,
            env.y,
            env.stops,
            env.n_stops,
            episode_reward,
            agent.method,
            agent.simplices,
            agent.vertices,
            agent.labels,
        )
    check = 0
    return env, agent, episode_reward


def plottour2(x, y, tour, coordinates, obj, simplices, vertices, labels):
    plt.scatter(x, y, c=labels)
    points = np.column_stack([x, y])
    for simplex in simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "c")
        plt.plot(
            points[vertices, 0],
            points[vertices, 1],
            "o",
            mec="r",
            color="none",
            lw=1,
            markersize=10,
        )
    count = 1
    for i in tour:
        if i == tour[-1]:
            k = str(i)
            g = str(tour[0])
            plt.arrow(
                coordinates[k][0],
                coordinates[k][1],
                diff(coordinates[k], coordinates[g])[0],
                diff(coordinates[k], coordinates[g])[1],
                head_width=0.15,
            )
            plt.text(coordinates[k][0], coordinates[k][1], s=str(count))

        else:
            k = str(i)
            g = str(tour[count])
            plt.text(coordinates[k][0], coordinates[k][1], s=str(count))
            plt.arrow(
                coordinates[k][0],
                coordinates[k][1],
                diff(coordinates[k], coordinates[g])[0],
                diff(coordinates[k], coordinates[g])[1],
                head_width=0.15,
            )
            count += 1
    plt.title("Gurobi-TSP-solver dist = " + str(round(obj, 2)))
    plt.show()


def test1(nodes, rep, max_box, methods):
    results = {}  # dict/ keys number of nodes/ value list of percentage diff btw diff methods and optimal solution

    for n_stops in nodes:
        # important functions used below

        # used to give tour length reward
        def actfun(x, n_stops, max_box):
            x = float(x)
            return 5 * (n_stops * max_box / 2 - x)

        # calc distance btwn points
        def distance(coordinates, city1, city2):
            c1 = coordinates[city1]
            c2 = coordinates[city2]
            diff = (c1[0] - c2[0], c1[1] - c2[1])
            return math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

        def diff(point1, point2):
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            return (dx, dy)

        def subtourelim(model, where):
            if where == GRB.Callback.MIPSOL:
                # make a list of edges selected in the solution
                vals = model.cbGetSolution(model._vars)
                selected = gp.tuplelist(
                    (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
                )
                # find the shortest cycle in the selected edge list
                tour = subtour(selected)
                if len(tour) < len(cities):
                    # add subtour elimination constr. for every pair of cities in subtour
                    model.cbLazy(
                        gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                        <= len(tour) - 1
                    )

        def subtour(edges):
            unvisited = cities[:]
            cycle = cities[:]  # Dummy - guaranteed to be replaced
            while unvisited:  # true if list is non-empty
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [
                        j for i, j in edges.select(current, "*") if j in unvisited
                    ]
                if len(thiscycle) <= len(cycle):
                    cycle = thiscycle  # New shortest subtour
            return cycle

        for k in range(rep):
            rep1 = k
            res_method = {i: [] for i in methods}
            check = {}
            print(
                "################ NODES =  "
                + str(n_stops)
                + " STEP "
                + str(k + 1)
                + " #########################"
            )
            xy = np.random.rand(n_stops, 2) * max_box  # generate points
            X = xy
            points = X
            # seperate coordinates
            x = xy[:, 0]
            y = xy[:, 1]
            coordinates = {}
            cities = []
            for i in range(n_stops):
                coordinates["city" + str(i + 1)] = (x[i], y[i])
                cities.append("city" + str(i + 1))

            dist = {
                (c1, c2): distance(coordinates, c1, c2)
                for c1, c2 in combinations(cities, 2)
            }

            # Gurobi code to get optimal solution
            m = gp.Model(name="TSP")
            vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name="x")
            # Symmetric direction: Copy the object
            for i, j in list(vars):
                vars[j, i] = vars[i, j]  # edge in opposite direction
            # Constraints: two edges incident to each city
            cons = m.addConstrs(vars.sum(c, "*") == 2 for c in cities)

            ############################################
            m._vars = vars
            m.Params.lazyConstraints = 1
            m.optimize(subtourelim)
            ##########################################
            # Retrieve solution

            vals = m.getAttr("x", vars)
            selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

            tour = subtour(selected)
            assert len(tour) == len(cities)

            # Create clusters using kmeans and convexhull

            wcss = []
            data = {}
            xy = pd.DataFrame(X)
            if n_stops < 11:
                range_clusters = n_stops
            else:
                range_clusters = 11
            for i in range(1, range_clusters):
                km = KMeans(n_clusters=i)
                km.fit_predict(xy)
                wcss.append(km.inertia_)
                data[i] = km.inertia_
            # Perform k-means clustering
            plt.figure()

            plt.plot(range(1, range_clusters), wcss)

            kn = KneeLocator(
                list(data.keys()),
                list(data.values()),
                curve="convex",
                direction="decreasing",
            )
            print(kn)
            k = kn.knee
            if k == None:
                k = 3
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)

            plt.figure()

            plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)

            mydict = {
                i: list(np.where(kmeans.labels_ == i)[0])
                for i in range(kmeans.n_clusters)
            }
            k = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            boundary_points = []
            boundary_index = []
            simplices = []
            vertices = []
            for i in range(kmeans.n_clusters):
                points = X[kmeans.labels_ == i]
                if len(points) < 3:
                    print(j)
                    boundary_index.extend(np.where(kmeans.labels_ == i)[0])

                    for g in points:
                        boundary_points.append(g)
                    continue
                hull = ConvexHull(points)

                boundary_index.extend(np.where(kmeans.labels_ == i)[0][hull.vertices])

                simplices.append(hull.simplices)
                vertices.append(hull.vertices)
                for simplex in hull.simplices:
                    j = mydict[i]
                    points = X[j]
                    plt.plot(points[simplex, 0], points[simplex, 1], "c")
                    plt.plot(
                        points[hull.vertices, 0],
                        points[hull.vertices, 1],
                        "o",
                        mec="r",
                        color="none",
                        lw=1,
                        markersize=10,
                    )
                    for g in hull.vertices.tolist():
                        boundary_points.append(points[g].tolist())

            boundary_points = {tuple(x) for x in boundary_points}
            boundary_index = list(boundary_index)
            labels = kmeans.labels_.tolist()
            n_cluster = kmeans.n_clusters

            plt.figure()
            count = 1
            plt.scatter(X[:, 0], X[:, 1], s=100, c=kmeans.labels_)
            # plt.plot(points[hull.vertices,0],points[hull.vertices,1],'o', mec='r', lw=1, markersize=10)

            # plot optimal solution with clusters
            for i in tour:
                if i == tour[-1]:
                    k = str(i)
                    g = str(tour[0])
                    plt.arrow(
                        coordinates[k][0],
                        coordinates[k][1],
                        diff(coordinates[k], coordinates[g])[0],
                        diff(coordinates[k], coordinates[g])[1],
                        head_width=0.05,
                    )
                    plt.text(coordinates[k][0], coordinates[k][1], s=str(count))

                else:
                    k = str(i)
                    g = str(tour[count])
                    plt.text(coordinates[k][0], coordinates[k][1], s=str(count))
                    plt.arrow(
                        coordinates[k][0],
                        coordinates[k][1],
                        diff(coordinates[k], coordinates[g])[0],
                        diff(coordinates[k], coordinates[g])[1],
                        head_width=0.05,
                    )
                    count += 1
            plt.title("Gurobi-TSP-solver dist = " + str(round(m.objVal, 2)))

            ep_res = [m.objVal]  # append result
            optimal_soln = m.objVal

            # RL Algorithm

            xy = X
            fixed = True
            env = Delivery(
                xy, boundary_index, n_stops=n_stops, max_box=max_box, fixed=fixed
            )
            plot1 = True

            n_episodes = 2000
            beta = 1
            alpha = 1
            for method in methods:
                agent = DeliveryQAgent(
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
                    env.observation_space,
                    env.action_space,
                )
                env, agent, episode_reward = run_n_episodes(
                    env,
                    agent,
                    plot1,
                    "training_" + str(env.n_stops) + "_stops.gif",
                    n_episodes,
                )
                ep_res.append(episode_reward)
                g = res_method[method]
                h = abs(optimal_soln - abs(episode_reward)) / optimal_soln
                g.append(h * 100)
                res_method[method] = g
            check[rep1] = ep_res
            # print(agent.visits)
        avg = []
        for l in methods:
            f = mean(res_method[l])
            avg.append(f)
        results[n_stops] = avg

    return results, check


r10, c10 = test1([10], 2, 20, methods)

"""
methods = ["cluster","epsilon","ucb","nearest neigh"]
plt.bar(methods,res.values())
plt.title("Compare with Optimal Soln nodes = 5")
plt.xlabel("Method")
plt.ylabel("% Diff")
plt.show()

r3 = {}
r3[0] =res[0]
r3[1] =res[1]
r3[2] =res[2]

methods = ["cluster","epsilon","ucb"]
plt.bar(methods,r3.values())
plt.title("Compare with Optimal Soln nodes = 5")
plt.xlabel("Method")
plt.ylabel("% Diff")
plt.show
"""
