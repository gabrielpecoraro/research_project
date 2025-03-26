# File: /tsp-project/tsp-project/src/utils/helpers.py

def calculate_distance(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return (dx**2 + dy**2) ** 0.5

def generate_random_points(num_points, max_x, max_y):
    import random
    return [(random.uniform(0, max_x), random.uniform(0, max_y)) for _ in range(num_points)]

def format_tour(tour):
    return " -> ".join(f"City {i+1}" for i in tour)