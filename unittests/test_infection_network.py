import unittest
import numpy as np
import networkx as nx
from modules.InfectionNetwork import (
    NodeState, 
    NetworkType, 
    InformationNetwork, 
    NetworkGenerator
)

class TestInfectionNetwork(unittest.TestCase):
    
    def test_node_state_enum(self):
        """Test that NodeState enum has correct values."""
        self.assertEqual(NodeState.SUSCEPTIBLE.value, 0)
        self.assertEqual(NodeState.INFECTED.value, 1)
    
    def test_network_types_enum(self):
        """Test that NetworkTypes enum has correct values."""
        self.assertEqual(NetworkType.ERDOS_RENYI.value, "erdos_renyi")
        self.assertEqual(NetworkType.SCALE_FREE.value, "scale_free")
        self.assertEqual(NetworkType.SMALL_WORLD.value, "small_world")
    
    def test_information_network_creation(self):
        """Test that InformationNetwork can be created."""
        network = InformationNetwork()
        self.assertIsInstance(network, InformationNetwork)
        self.assertIsInstance(network, nx.DiGraph)
    
    def test_information_network_add_node(self):
        """Test that nodes can be added with default attributes."""
        network = InformationNetwork()
        network.add_node(1)
        
        # Check that node exists and has default attributes
        self.assertIn(1, network.nodes)
        self.assertEqual(network.nodes[1]['state'], NodeState.SUSCEPTIBLE)
        self.assertGreaterEqual(network.nodes[1]['skepticism'], 0.0)
        self.assertLessEqual(network.nodes[1]['skepticism'], 1.0)
        self.assertEqual(network.nodes[1]['length_of_infection'], 0)
    
    def test_information_network_add_node_with_attributes(self):
        """Test that nodes can be added with custom attributes."""
        network = InformationNetwork()
        network.add_node(1, state=NodeState.INFECTED, skepticism=0.8, length_of_infection=5)
        
        # Check that node exists and has custom attributes
        self.assertIn(1, network.nodes)
        self.assertEqual(network.nodes[1]['state'], NodeState.INFECTED)
        self.assertEqual(network.nodes[1]['skepticism'], 0.8)
        self.assertEqual(network.nodes[1]['length_of_infection'], 5)
    
    def test_information_network_add_edge(self):
        """Test that edges can be added with default weight."""
        network = InformationNetwork()
        network.add_node(1)
        network.add_node(2)
        network.add_edge(1, 2)
        
        # Check that edge exists and has default weight
        self.assertIn((1, 2), network.edges)
        self.assertEqual(network.edges[1, 2]['weight'], 1.0)
    
    def test_information_network_add_edge_with_weight(self):
        """Test that edges can be added with custom weight."""
        network = InformationNetwork()
        network.add_node(1)
        network.add_node(2)
        network.add_edge(1, 2, weight=0.5)
        
        # Check that edge exists and has custom weight
        self.assertIn((1, 2), network.edges)
        self.assertEqual(network.edges[1, 2]['weight'], 0.5)
    
    def test_erdos_renyi_generator(self):
        """Test that Erdos-Renyi network can be generated."""
        n = 100
        p = 0.1
        network = NetworkGenerator.erdos_renyi(n, p)
        
        # Check that network has correct size and type
        self.assertEqual(len(network.nodes), n)
        self.assertEqual(network.type, NetworkType.ERDOS_RENYI)
        
        # Check that all nodes have required attributes
        for node in network.nodes:
            self.assertIn('state', network.nodes[node])
            self.assertIn('skepticism', network.nodes[node])
            self.assertIn('infection_length', network.nodes[node])
    
    def test_scale_free_generator(self):
        """Test that Scale-Free network can be generated."""
        n = 100
        m = 3
        network = NetworkGenerator.scale_free(n, m)
        
        # Check that network has correct size and type
        self.assertEqual(len(network.nodes), n)
        self.assertEqual(network.type, NetworkType.SCALE_FREE)
        
        # Check that all nodes have required attributes
        for node in network.nodes:
            self.assertIn('state', network.nodes[node])
            self.assertIn('skepticism', network.nodes[node])
            self.assertIn('infection_length', network.nodes[node])
    
    def test_small_world_generator(self):
        """Test that Small-World network can be generated."""
        n = 100
        k = 4
        p = 0.1
        network = NetworkGenerator.small_world(n, k, p)
        
        # Check that network has correct size and type
        self.assertEqual(len(network.nodes), n)
        self.assertEqual(network.type, NetworkType.SMALL_WORLD)
        
        # Check that all nodes have required attributes
        for node in network.nodes:
            self.assertIn('state', network.nodes[node])
            self.assertIn('skepticism', network.nodes[node])
            self.assertIn('infection_length', network.nodes[node])
    
    def test_edge_weights(self):
        """Test that edges have weights in generated networks."""
        n = 50
        p = 0.1
        network = NetworkGenerator.erdos_renyi(n, p)
        
        # Check that all edges have weight attribute
        for u, v in network.edges:
            self.assertIn('weight', network.edges[u, v])
            self.assertGreaterEqual(network.edges[u, v]['weight'], 0.0)
            self.assertLessEqual(network.edges[u, v]['weight'], 1.0)

if __name__ == '__main__':
    unittest.main()
