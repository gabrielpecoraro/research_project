import unittest
import numpy as np
from src.environment.delivery import Delivery

class TestDelivery(unittest.TestCase):

    def setUp(self):
        self.xy = np.array([[0, 0], [1, 1], [2, 2]])
        self.delivery_env = Delivery(xy=self.xy, boundary_index=[0, 1], n_stops=3, max_box=10, fixed=False)

    def test_generate_stops(self):
        self.delivery_env._generate_stops()
        self.assertEqual(len(self.delivery_env.stops), self.delivery_env.n_stops)

    def test_reset(self):
        initial_stop = self.delivery_env.reset()
        self.assertIn(initial_stop, range(self.delivery_env.n_stops))

    def test_step(self):
        self.delivery_env.reset()
        new_state, reward = self.delivery_env.step(1)
        self.assertEqual(new_state, 1)
        self.assertIsInstance(reward, (int, float))

    def test_distance(self):
        dist = self.delivery_env.distance(self.xy, 0, 1)
        self.assertAlmostEqual(dist, np.sqrt(2), places=5)

if __name__ == '__main__':
    unittest.main()