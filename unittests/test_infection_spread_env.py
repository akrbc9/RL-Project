import unittest
import numpy as np
import networkx as nx
import sys,os
from gymnasium import spaces

# Add the parent directory to the path so modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from InfectionNetwork import NodeState, InformationNetwork
from NetworkEnvironmentConfig import NetworkEnvironmentConfig
from Features import FeatureRegistry
from InfectionSpreadEnv import MultiAgentNetworkEnv, env_creator, InfoSpreadCallbacks

class TestInfectionSpreadEnv(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple test environment."""
        # Reset feature registry to ensure consistent feature count
        FeatureRegistry.reset()
        
        # Register features to match what's expected in the environment
        FeatureRegistry.register("state", None)
        FeatureRegistry.register("skepticism", None)
        FeatureRegistry.register("length_of_infection", None)
        FeatureRegistry.register("internal_influence", None)
        FeatureRegistry.register("infection_pressure", None)
        FeatureRegistry.register("spreading_potential", None)
        
        # Create a test config
        self.config = NetworkEnvironmentConfig(
            network_type="scale_free",
            nx_n=20,  # Small network for testing
            nx_m=2,
            num_nodes=20,
            num_features=FeatureRegistry.get_feature_count(),
            infection_probability=0.5,
            initial_infection_density=0.2,
            recovery_probability=0.3,
            max_steps=50,
            max_action_radius=1
        )
        
        # Create environment
        self.env = MultiAgentNetworkEnv(self.config)
    
    def test_env_initialization(self):
        """Test that environment can be initialized."""
        self.assertIsInstance(self.env, MultiAgentNetworkEnv)
        self.assertEqual(self.env.num_nodes, 20)
        self.assertEqual(self.env.max_steps, 50)
        self.assertEqual(self.env.max_action_radius, 1)
        self.assertEqual(self.env.infection_prob, 0.5)
        self.assertEqual(self.env.recovery_prob, 0.3)
        
        # Check network
        self.assertIsInstance(self.env.network, InformationNetwork)
        self.assertEqual(len(self.env.network.nodes), 20)
        
        # Check action spaces
        self.assertIn("agent_a", self.env.action_space)
        self.assertIn("agent_b", self.env.action_space)
        
        # Check observation spaces
        self.assertIn("agent_a", self.env.observation_space)
        self.assertIn("agent_b", self.env.observation_space)
    
    def test_env_reset(self):
        """Test that environment can be reset."""
        obs, info = self.env.reset()
        
        # Check observations
        self.assertIn("agent_a", obs)
        self.assertIn("agent_b", obs)
        self.assertIsInstance(obs["agent_a"], np.ndarray)
        self.assertIsInstance(obs["agent_b"], np.ndarray)
        
        # Check shape of observations
        self.assertEqual(obs["agent_a"].shape, (20, 6))  # 20 nodes, 6 features
        self.assertEqual(obs["agent_b"].shape, (20, 6))  # 20 nodes, 6 features
        
        # Check that info is empty dict
        self.assertEqual(info, {})
        
        # Check that there are infected nodes after reset (initial infection)
        infected_count = sum(1 for n, d in self.env.network.nodes(data=True) 
                           if d['state'] == NodeState.INFECTED)
        expected_count = int(self.config.initial_infection_density * self.config.num_nodes)
        self.assertEqual(infected_count, expected_count)
    
    def test_env_step_agent_a(self):
        """Test that agent A can take actions."""
        # Reset environment
        self.env.reset()
        
        # Prepare action for agent A (infect node 0)
        actions = {
            "agent_a": {"node_id": 0}
        }
        
        # Take step with just agent A
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        
        # Check observations
        self.assertIn("agent_a", obs)
        self.assertIn("agent_b", obs)
        
        # Check rewards
        self.assertIn("agent_a", rewards)
        self.assertIn("agent_b", rewards)
        
        # Check terminateds
        self.assertIn("agent_a", terminateds)
        self.assertIn("agent_b", terminateds)
        self.assertIn("__all__", terminateds)
        
        # Check truncateds
        self.assertIn("agent_a", truncateds)
        self.assertIn("agent_b", truncateds)
        self.assertIn("__all__", truncateds)
        
        # Check infos
        self.assertIn("agent_a", infos)
        self.assertIn("agent_b", infos)
    
    def test_env_step_agent_b(self):
        """Test that agent B can take actions."""
        # Reset environment
        self.env.reset()
        
        # Set node 0 to infected for testing healing
        self.env.network.nodes[0]['state'] = NodeState.INFECTED
        
        # Prepare action for agent B (heal node 0)
        actions = {
            "agent_b": {"node_id": 0, "radius": 1}
        }
        
        # Take step with just agent B
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        
        # Check that step was taken correctly
        self.assertIsInstance(obs["agent_a"], np.ndarray)
        self.assertIsInstance(obs["agent_b"], np.ndarray)
        
        # Check that agent B got a reward
        self.assertIsInstance(rewards["agent_b"], float)
    
    def test_env_step_both_agents(self):
        """Test that both agents can take actions together."""
        # Reset environment
        self.env.reset()
        
        # Prepare actions for both agents
        actions = {
            "agent_a": {"node_id": 0},
            "agent_b": {"node_id": 1, "radius": 1}
        }
        
        # Take step with both agents
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        
        # Check that step was taken correctly
        self.assertIsInstance(obs["agent_a"], np.ndarray)
        self.assertIsInstance(obs["agent_b"], np.ndarray)
        
        # Check that both agents got rewards
        self.assertIsInstance(rewards["agent_a"], float)
        self.assertIsInstance(rewards["agent_b"], float)
    
    def test_infection_propagation(self):
        """Test that infection propagates through the network."""
        # Reset environment
        self.env.reset()
        
        # Set all nodes to susceptible first
        for node in self.env.network.nodes:
            self.env.network.nodes[node]['state'] = NodeState.SUSCEPTIBLE
        
        # Set node 0 to infected
        self.env.network.nodes[0]['state'] = NodeState.INFECTED
        
        # Create a direct edge from node 0 to node 1 with high weight
        if (0, 1) not in self.env.network.edges:
            self.env.network.add_edge(0, 1, weight=1.0)
        else:
            self.env.network.edges[0, 1]['weight'] = 1.0
        
        # Force infection probability to 1.0 for this test
        original_prob = self.env.infection_prob
        self.env.infection_prob = 1.0
        
        # Reset skepticism to 0 for this test
        self.env.network.nodes[1]['skepticism'] = 0.0
        
        # Propagate information
        newly_infected = self.env._propagate_information()
        
        # Check that node 1 got infected
        self.assertIn(1, newly_infected)
        self.assertEqual(self.env.network.nodes[1]['state'], NodeState.INFECTED)
        
        # Restore original infection probability
        self.env.infection_prob = original_prob
    
    def test_recovery_process(self):
        """Test that nodes can recover from infection."""
        # Reset environment
        self.env.reset()
        
        # Set node 0 to infected
        self.env.network.nodes[0]['state'] = NodeState.INFECTED
        
        # Force recovery probability to 1.0 for this test
        original_prob = self.env.recovery_prob
        self.env.recovery_prob = 1.0
        
        # Process recovery
        newly_recovered = self.env._process_recovery()
        
        # Check that node 0 recovered
        self.assertIn(0, newly_recovered)
        self.assertEqual(self.env.network.nodes[0]['state'], NodeState.SUSCEPTIBLE)
        
        # Restore original recovery probability
        self.env.recovery_prob = original_prob
    
    def test_observation_space_contains(self):
        """Test the observation_space_contains method."""
        # Reset environment
        obs, _ = self.env.reset()
        
        # Check that observation is contained in observation space
        self.assertTrue(self.env.observation_space_contains(obs))
        
        # Check that invalid observation is not contained
        invalid_obs = {"agent_a": np.zeros((10, 3))}  # Wrong shape
        self.assertFalse(self.env.observation_space_contains(invalid_obs))
    
    def test_action_space_contains(self):
        """Test the action_space_contains method."""
        # Valid actions
        valid_actions = {
            "agent_a": {"node_id": 0},
            "agent_b": {"node_id": 1, "radius": 1}
        }
        
        # Check that valid actions are contained in action space
        self.assertTrue(self.env.action_space_contains(valid_actions))
        
        # Invalid actions
        invalid_actions = {
            "agent_a": {"node_id": 100},  # Node ID out of range
            "agent_b": {"node_id": 1, "radius": 5}  # Radius out of range
        }
        
        # Check that invalid actions are not contained
        self.assertFalse(self.env.action_space_contains(invalid_actions))
    
    def test_get_infected_count(self):
        """Test that infected count is calculated correctly."""
        # Reset environment
        self.env.reset()
        
        # Set all nodes to susceptible
        for node in self.env.network.nodes:
            self.env.network.nodes[node]['state'] = NodeState.SUSCEPTIBLE
        
        # Set specific nodes to infected
        infected_nodes = [0, 1, 2]
        for node in infected_nodes:
            self.env.network.nodes[node]['state'] = NodeState.INFECTED
        
        # Check infected count
        infected_count = self.env._get_infected_count()
        self.assertEqual(infected_count, len(infected_nodes))
    
    def test_get_susceptible_count(self):
        """Test that susceptible count is calculated correctly."""
        # Reset environment
        self.env.reset()
        
        # Set all nodes to infected
        for node in self.env.network.nodes:
            self.env.network.nodes[node]['state'] = NodeState.INFECTED
        
        # Set specific nodes to susceptible
        susceptible_nodes = [0, 1, 2]
        for node in susceptible_nodes:
            self.env.network.nodes[node]['state'] = NodeState.SUSCEPTIBLE
        
        # Check susceptible count
        susceptible_count = self.env._get_susceptible_count()
        self.assertEqual(susceptible_count, len(susceptible_nodes))
    
    def test_infect_node(self):
        """Test that nodes can be infected."""
        # Reset environment
        self.env.reset()
        
        # Set node 0 to susceptible
        self.env.network.nodes[0]['state'] = NodeState.SUSCEPTIBLE
        
        # Infect node with 100% probability
        success = self.env._infect_node(0, 1.0)
        
        # Check that infection was successful
        self.assertTrue(success)
        self.assertEqual(self.env.network.nodes[0]['state'], NodeState.INFECTED)
        
        # Try to infect an already infected node
        success = self.env._infect_node(0, 1.0)
        
        # Check that infection failed (already infected)
        self.assertFalse(success)
    
    def test_heal_node(self):
        """Test that nodes can be healed."""
        # Reset environment
        self.env.reset()
        
        # Set node 0 to infected
        self.env.network.nodes[0]['state'] = NodeState.INFECTED
        self.env.network.nodes[0]['length_of_infection'] = 5
        
        # Heal node with 100% probability
        success = self.env._heal_node(0, 1.0)
        
        # Check that healing was successful
        self.assertTrue(success)
        self.assertEqual(self.env.network.nodes[0]['state'], NodeState.SUSCEPTIBLE)
        self.assertEqual(self.env.network.nodes[0]['length_of_infection'], 0)
        
        # Try to heal an already susceptible node
        success = self.env._heal_node(0, 1.0)
        
        # Check that healing failed (already susceptible)
        self.assertFalse(success)
    
    def test_env_creator(self):
        """Test the env_creator function."""
        # Create a config
        config = self.config.to_dict()
        
        # Create environment using env_creator
        env = env_creator(config)
        
        # Check that environment was created correctly
        self.assertIsInstance(env, MultiAgentNetworkEnv)
        self.assertEqual(env.num_nodes, 20)

if __name__ == '__main__':
    unittest.main()