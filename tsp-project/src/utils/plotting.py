# filepath: /tsp-project/tsp-project/src/utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_delivery_stops(xy, stops):
    plt.figure(figsize=(10, 6))
    plt.scatter(xy[:, 0], xy[:, 1], c='red', label='Delivery Stops')
    for i, stop in enumerate(stops):
        plt.annotate(f'Stop {i+1}', (xy[stop, 0], xy[stop, 1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Delivery Stops')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()

def plot_tour(xy, tour):
    plt.figure(figsize=(10, 6))
    plt.plot(xy[tour, 0], xy[tour, 1], marker='o', color='blue', label='Tour Path')
    plt.title('Tour Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()