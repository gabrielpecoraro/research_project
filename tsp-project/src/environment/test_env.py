# Test script to visualize the grid environment
import numpy as np
from delivery import Delivery

# Generate random coordinates along streets
grid_size = 5
block_size = 4
n_stops = 5
max_box = grid_size * block_size


# Generate coordinates only on streets
def generate_street_coordinates(n_stops, grid_size, block_size):
    coords = []
    while len(coords) < n_stops:
        x = np.random.randint(0, grid_size * block_size)
        y = np.random.randint(0, grid_size * block_size)
        # Check if coordinate is on a street
        if x % block_size == 0 or y % block_size == 0:
            coords.append([x, y])
    return np.array(coords)


xy = generate_street_coordinates(n_stops, grid_size, block_size)

# Create environment
env = Delivery(
    xy=xy,
    boundary_index=list(range(n_stops)),
    n_stops=n_stops,
    max_box=max_box,
    fixed=False,
    grid_size=grid_size,
    block_size=block_size,
)

# Render environment
env.render()
