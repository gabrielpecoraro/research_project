# File: /tsp-project/tsp-project/tests/test_agent.py

import unittest
from src.agents.delivery_q_agent import DeliveryQAgent
from src.environment.delivery import Delivery

class TestDeliveryQAgent(unittest.TestCase):

    def setUp(self):
        # Setup the environment and agent for testing
        self.xy = [[0, 0], [1, 1], [2, 2]]  # Example coordinates
        self.max_box = 10
        self.method = "epsilon"
        self.boundary_index = [0, 1]
        self.boundary_points = []
        self.mydict = {}
        self.labels = []
        self.n_cluster = 1
        self.simplices = []
        self.vertices = []
        self.states_size = 3
        self.actions_size = 3
        self.agent = DeliveryQAgent(
            xy=self.xy,
            max_box=self.max_box,
            method=self.method,
            boundary_index=self.boundary_index,
            boundary_points=self.boundary_points,
            mydict=self.mydict,
            labels=self.labels,
            n_cluster=self.n_cluster,
            simplices=self.simplices,
            vertices=self.vertices,
            states_size=self.states_size,
            actions_size=self.actions_size
        )

    def test_agent_initialization(self):
        self.assertEqual(self.agent.method, self.method)
        self.assertEqual(self.agent.max_box, self.max_box)
        self.assertEqual(len(self.agent.xy), len(self.xy))

    def test_agent_action_selection(self):
        action = self.agent.act(0)
        self.assertIn(action, range(self.actions_size))

    def test_agent_training(self):
        initial_q = self.agent.Q.copy()
        self.agent.train(0, 1, 10)
        self.assertNotEqual(initial_q[0, 1], self.agent.Q[0, 1])

if __name__ == '__main__':
    unittest.main()