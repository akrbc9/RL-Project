import unittest
import numpy as np
import networkx as nx
import sys,os
from gymnasium import spaces

# Add the parent directory to the path so modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modules.InfectionNetwork import NodeState, InformationNetwork
from modules.Features import (
    FeatureRegistry, 
    NodeFeature, 
    FeatureExtractors, 
    NodeFeatures
)

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple test network."""
        FeatureRegistry.reset()
        
        # Create a small test network
        self.network = InformationNetwork()
        
        # Add nodes
        for i in range(5):
            self.network.add_node(i)
        
        # Set some nodes to infected
        self.network.nodes[0]['state'] = NodeState.INFECTED
        self.network.nodes[1]['state'] = NodeState.INFECTED
        self.network.nodes[0]['length_of_infection'] = 3
        self.network.nodes[1]['length_of_infection'] = 2
        
        # Add edges
        self.network.add_edge(0, 2, weight=0.8)
        self.network.add_edge(1, 3, weight=0.6)
        self.network.add_edge(2, 4, weight=0.7)
        self.network.add_edge(0, 1, weight=0.9)
    
    def test_feature_registry_reset(self):
        """Test that FeatureRegistry can be reset."""
        # Register a feature
        FeatureRegistry.register("test_feature", None)
        
        # Check that feature is registered
        self.assertIn("test_feature", FeatureRegistry._feature_indices)
        
        # Reset registry
        FeatureRegistry.reset()
        
        # Check that feature is no longer registered
        self.assertNotIn("test_feature", FeatureRegistry._feature_indices)
        self.assertEqual(FeatureRegistry._next_index, 0)
    
    def test_feature_registry_register(self):
        """Test that features can be registered with the registry."""
        # Reset registry
        FeatureRegistry.reset()
        
        # Register features
        index1 = FeatureRegistry.register("feature1", None)
        index2 = FeatureRegistry.register("feature2", None)
        
        # Check indices
        self.assertEqual(index1, 0)
        self.assertEqual(index2, 1)
        
        # Check features are registered
        self.assertIn("feature1", FeatureRegistry._feature_indices)
        self.assertIn("feature2", FeatureRegistry._feature_indices)
    
    def test_feature_registry_get_index(self):
        """Test that feature indices can be retrieved."""
        # Reset registry
        FeatureRegistry.reset()
        
        # Register features
        FeatureRegistry.register("feature1", None)
        FeatureRegistry.register("feature2", None)
        
        # Get indices
        index1 = FeatureRegistry.get_index("feature1")
        index2 = FeatureRegistry.get_index("feature2")
        
        # Check indices
        self.assertEqual(index1, 0)
        self.assertEqual(index2, 1)
        
        # Check getting non-existent feature raises error
        with self.assertRaises(ValueError):
            FeatureRegistry.get_index("non_existent_feature")
    
    def test_feature_registry_get_all_indices(self):
        """Test that all feature indices can be retrieved."""
        # Reset registry
        FeatureRegistry.reset()
        
        # Register features
        FeatureRegistry.register("feature1", None)
        FeatureRegistry.register("feature2", None)
        
        # Get all indices
        indices = FeatureRegistry.get_all_indices()
        
        # Check indices
        self.assertEqual(indices, {"feature1": 0, "feature2": 1})
    
    def test_feature_registry_get_feature_count(self):
        """Test that feature count can be retrieved."""
        # Reset registry
        FeatureRegistry.reset()
        
        # Register features
        FeatureRegistry.register("feature1", None)
        FeatureRegistry.register("feature2", None)
        
        # Get feature count
        count = FeatureRegistry.get_feature_count()
        
        # Check count
        self.assertEqual(count, 2)
    
    def test_node_feature_creation(self):
        """Test that NodeFeature can be created and registered."""
        # Reset registry
        FeatureRegistry.reset()
        
        # Create a feature
        feature = NodeFeature(name="test_feature", typeription="Test feature", extractor=None)
        
        # Check feature properties
        self.assertEqual(feature.name, "test_feature")
        self.assertEqual(feature.typeription, "Test feature")
        self.assertIsNone(feature.extractor)
        
        # Check feature is registered
        self.assertIn("test_feature", FeatureRegistry._feature_indices)
        self.assertEqual(feature.index, 0)
    
    def test_feature_extractor_get_state(self):
        """Test that state feature extractor works correctly."""
        # Extract states
        states = FeatureExtractors.get_state(self.network)
        
        # Check states
        self.assertEqual(states[0], NodeState.INFECTED.value)
        self.assertEqual(states[1], NodeState.INFECTED.value)
        self.assertEqual(states[2], NodeState.SUSCEPTIBLE.value)
        self.assertEqual(states[3], NodeState.SUSCEPTIBLE.value)
        self.assertEqual(states[4], NodeState.SUSCEPTIBLE.value)
    
    def test_feature_extractor_get_skepticism(self):
        """Test that skepticism feature extractor works correctly."""
        # Set skepticism values
        self.network.nodes[0]['skepticism'] = 0.2
        self.network.nodes[1]['skepticism'] = 0.4
        
        # Extract skepticism
        skepticism = FeatureExtractors.get_skepticism(self.network)
        
        # Check skepticism
        self.assertEqual(skepticism[0], 0.2)
        self.assertEqual(skepticism[1], 0.4)
    
    def test_feature_extractor_get_length_of_infection(self):
        """Test that length of infection feature extractor works correctly."""
        # Extract length of infection
        length = FeatureExtractors.get_length_of_infection(self.network)
        
        # Check length of infection (normalized)
        max_length = 3  # Node 0 has length 3
        self.assertEqual(length[0], 3 / max_length)
        self.assertEqual(length[1], 2 / max_length)
        self.assertEqual(length[2], 0.0)
        self.assertEqual(length[3], 0.0)
        self.assertEqual(length[4], 0.0)
    
    def test_feature_extractor_get_infection_pressure(self):
        """Test that infection pressure feature extractor works correctly."""
        # Extract infection pressure
        pressure = FeatureExtractors.get_infection_pressure(self.network)
        
        # Check pressure (node 2 is influenced by infected node 0)
        self.assertGreater(pressure[2], 0.0)
        self.assertLessEqual(pressure[2], 1.0)
        
        # Nodes that aren't susceptible or don't have infected neighbors should have 0 pressure
        self.assertEqual(pressure[0], 0.0)  # Already infected
        self.assertEqual(pressure[1], 0.0)  # Already infected
        self.assertEqual(pressure[4], 0.0)  # No infected neighbors
    
    def test_feature_registry_extract_feature_matrix(self):
        """Test that feature matrix can be extracted."""
        # Reset registry
        FeatureRegistry.reset()
        
        # Register test features
        FeatureRegistry.register("state", FeatureExtractors.get_state)
        FeatureRegistry.register("skepticism", FeatureExtractors.get_skepticism)
        
        # Extract feature matrix
        feature_matrix, node_ids = FeatureRegistry.extract_feature_matrix(self.network)
        
        # Check matrix shape (nodes x features)
        self.assertEqual(feature_matrix.shape, (5, 2))
        
        # Check node order
        self.assertEqual(len(node_ids), 5)
        
        # Check feature values
        node_indices = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Check state feature
        state_idx = FeatureRegistry.get_index("state")
        self.assertEqual(feature_matrix[node_indices[0], state_idx], NodeState.INFECTED.value)
        self.assertEqual(feature_matrix[node_indices[2], state_idx], NodeState.SUSCEPTIBLE.value)
        
        # Check skepticism feature
        skepticism_idx = FeatureRegistry.get_index("skepticism")
        for node_id in range(5):
            matrix_idx = node_indices[node_id]
            self.assertEqual(feature_matrix[matrix_idx, skepticism_idx], 
                             self.network.nodes[node_id]['skepticism'])
    
    def test_node_features_class(self):
        """Test that NodeFeatures class registers features correctly."""
        # Reset registry first
        FeatureRegistry.reset()
        
        # Access NodeFeatures class to trigger registration
        features = NodeFeatures()
        
        # Check registered features
        registered_features = FeatureRegistry.get_all_indices()
        self.assertIn("state", registered_features)
        self.assertIn("skepticism", registered_features)
        self.assertIn("length_of_infection", registered_features)
        self.assertIn("internal_influence", registered_features)
        self.assertIn("infection_pressure", registered_features)
        self.assertIn("spreading_potential", registered_features)
        
        # Check get_all method
        all_features = NodeFeatures.get_all()
        self.assertEqual(len(all_features), 6)  # 6 features defined in class

if __name__ == '__main__':
    unittest.main()
