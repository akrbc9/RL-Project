import unittest
import numpy as np
import sys,os
from gymnasium import spaces

# Add the parent directory to the path so modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modules.NetworkEnvironmentConfig import NetworkEnvironmentConfig
from modules.InfectionNetwork import InformationNetwork
from modules.Features import FeatureRegistry

class TestNetworkEnvironmentConfig(unittest.TestCase):
    
    def setUp(self):
        """Set up a default config for testing."""
        # Reset feature registry to ensure consistent feature count
        FeatureRegistry.reset()
        
        # Register a couple of test features
        FeatureRegistry.register("test_feature1", None)
        FeatureRegistry.register("test_feature2", None)
        
        # Create a default config
        self.config = NetworkEnvironmentConfig(
            network_type="scale_free",
            nx_n=100,
            nx_m=3,
            nx_k=5,  # For small-world
            nx_p=0.1,  # For small-world and Erdos-Renyi
            num_nodes=100,
            num_features=FeatureRegistry.get_feature_count(),
            infection_probability=0.5,
            initial_infection_density=0.2,
            recovery_probability=0.3,
            max_steps=100,
            max_action_radius=2
        )
    
    def test_config_initialization(self):
        """Test that config can be initialized with correct values."""
        self.assertEqual(self.config.network_type, "scale_free")
        self.assertEqual(self.config.nx_n, 100)
        self.assertEqual(self.config.nx_m, 3)
        self.assertEqual(self.config.nx_k, 5)
        self.assertEqual(self.config.nx_p, 0.1)
        self.assertEqual(self.config.num_nodes, 100)
        self.assertEqual(self.config.num_features, 2)  # From setUp
        self.assertEqual(self.config.infection_probability, 0.5)
        self.assertEqual(self.config.initial_infection_density, 0.2)
        self.assertEqual(self.config.recovery_probability, 0.3)
        self.assertEqual(self.config.max_steps, 100)
        self.assertEqual(self.config.max_action_radius, 2)
    
    def test_get_information_network_scale_free(self):
        """Test that scale-free network can be generated."""
        network = self.config.get_information_network()
        self.assertIsInstance(network, InformationNetwork)
        self.assertEqual(network.type, "scale_free")
        self.assertEqual(len(network.nodes), self.config.nx_n)
    
    def test_get_information_network_erdos_renyi(self):
        """Test that Erdos-Renyi network can be generated."""
        self.config.network_type = "erdos_renyi"
        network = self.config.get_information_network()
        self.assertIsInstance(network, InformationNetwork)
        self.assertEqual(network.type, "erdos_renyi")
        self.assertEqual(len(network.nodes), self.config.nx_n)
    
    def test_get_information_network_small_world(self):
        """Test that small-world network can be generated."""
        self.config.network_type = "small_world"
        network = self.config.get_information_network()
        self.assertIsInstance(network, InformationNetwork)
        self.assertEqual(network.type, "small_world")
        self.assertEqual(len(network.nodes), self.config.nx_n)
    
    def test_get_information_network_invalid_type(self):
        """Test that invalid network type raises error."""
        self.config.network_type = "invalid_type"
        with self.assertRaises(ValueError):
            self.config.get_information_network()
    
    def test_get_action_space(self):
        """Test that action spaces are correctly generated."""
        action_spaces = self.config.get_action_space()
        
        # Check agent_a action space
        self.assertIn("agent_a", action_spaces)
        self.assertIsInstance(action_spaces["agent_a"], spaces.Dict)
        self.assertIn("node_id", action_spaces["agent_a"].spaces)
        self.assertIsInstance(action_spaces["agent_a"].spaces["node_id"], spaces.Discrete)
        self.assertEqual(action_spaces["agent_a"].spaces["node_id"].n, self.config.num_nodes)
        
        # Check agent_b action space
        self.assertIn("agent_b", action_spaces)
        self.assertIsInstance(action_spaces["agent_b"], spaces.Dict)
        self.assertIn("node_id", action_spaces["agent_b"].spaces)
        self.assertIn("radius", action_spaces["agent_b"].spaces)
        self.assertIsInstance(action_spaces["agent_b"].spaces["node_id"], spaces.Discrete)
        self.assertIsInstance(action_spaces["agent_b"].spaces["radius"], spaces.Discrete)
        self.assertEqual(action_spaces["agent_b"].spaces["node_id"].n, self.config.num_nodes)
        self.assertEqual(action_spaces["agent_b"].spaces["radius"].n, self.config.max_action_radius + 1)
    
    def test_get_observation_space(self):
        """Test that observation spaces are correctly generated."""
        observation_spaces = self.config.get_observation_space()
        
        # Check agent_a observation space
        self.assertIn("agent_a", observation_spaces)
        self.assertIsInstance(observation_spaces["agent_a"], spaces.Box)
        self.assertEqual(observation_spaces["agent_a"].shape, (self.config.num_nodes, self.config.num_features))
        
        # Check agent_b observation space
        self.assertIn("agent_b", observation_spaces)
        self.assertIsInstance(observation_spaces["agent_b"], spaces.Box)
        self.assertEqual(observation_spaces["agent_b"].shape, (self.config.num_nodes, self.config.num_features))
    
    def test_to_dict(self):
        """Test that config can be converted to dictionary."""
        config_dict = self.config.to_dict()
        
        # Check network data
        self.assertIn('network_data', config_dict)
        self.assertEqual(config_dict['network_data']['network_type'], "scale_free")
        self.assertEqual(config_dict['network_data']['nx_n'], 100)
        self.assertEqual(config_dict['network_data']['nx_m'], 3)
        
        # Check environment parameters
        self.assertEqual(config_dict['num_nodes'], 100)
        self.assertEqual(config_dict['num_features'], 2)
        self.assertEqual(config_dict['infection_probability'], 0.5)
        self.assertEqual(config_dict['initial_infection_density'], 0.2)
        self.assertEqual(config_dict['recovery_probability'], 0.3)
        self.assertEqual(config_dict['max_steps'], 100)
        self.assertEqual(config_dict['max_action_radius'], 2)
    
    def test_from_dict(self):
        """Test that config can be reconstructed from dictionary."""
        # Convert to dict and back
        config_dict = self.config.to_dict()
        new_config = NetworkEnvironmentConfig.from_dict(config_dict)
        
        # Check reconstructed config
        self.assertEqual(new_config.network_type, "scale_free")
        self.assertEqual(new_config.nx_n, 100)
        self.assertEqual(new_config.nx_m, 3)
        self.assertEqual(new_config.nx_k, 5)
        self.assertEqual(new_config.nx_p, 0.1)
        self.assertEqual(new_config.num_nodes, 100)
        self.assertEqual(new_config.num_features, 2)
        self.assertEqual(new_config.infection_probability, 0.5)
        self.assertEqual(new_config.initial_infection_density, 0.2)
        self.assertEqual(new_config.recovery_probability, 0.3)
        self.assertEqual(new_config.max_steps, 100)
        self.assertEqual(new_config.max_action_radius, 2)

if __name__ == '__main__':
    unittest.main()
