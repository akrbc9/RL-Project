import numpy as np
import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Union
import gym
from gym import spaces


class NodeState(Enum):
    """Enum for the possible states of a node in the SIR model."""
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2


class InformationNetwork(nx.DiGraph):
    """Extended directed graph for information spread modeling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_node(self, node_for_adding, **attr):
        """Add a node with SIR state attribute."""
        if 'state' not in attr:
            attr['state'] = NodeState.SUSCEPTIBLE
        super().add_node(node_for_adding, **attr)
    
    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge with weight attribute."""
        if 'weight' not in attr:
            attr['weight'] = 1.0 # Default weight: 1 
        super().add_edge(u_of_edge, v_of_edge, **attr)
    
    # def get_node_features(self, node_id):
    #     """
    #     Extract features for a specific node.
    #     Args:
    #         node_id: ID of target node. 
    #                  If none, raises ValueError
    #     """
    #     if node_id is not None: 
    #         if node_id not in self: 
    #             raise ValueError(f"Node {node_id} not found in network.")
        
    #         node_data = self.nodes[node_id]       

    #         # Collect all relevant features  
    #         state_value = node_data['state'].value / 2.0 # Normalize to [0,1] here
    #         in_degree = self.in_degree(node_id, weight='weight')/len(self) # Divide by number of nodes for scale invariance 
    #         out_degree = self.out_degree(node_id, weight='weight')/len(self)

    #         # Other useful features: centrality, clustering. 
    #         return np.array([state_value,in_degree,out_degree])

    #     else:
    #         raise ValueError(f"Node ID not provided in get_node_features.")
    
    # def get_network_features(self):
    #     """Extract global network features."""
    #     # To be implemented: extract network-level features for RL state
    #     # Features include: .... TODO: Make List of Network features to extract. 

    #     # Prevalence features
    #     num_nodes = len(self)
    #     num_infected = len([node_data['state'] == NodeState.INFECTED for node_data in self.nodes])
    #     num_susceptible = num_nodes - num_infected

    
    def visualize(self):
        """Visualize the current state of the network."""
        # Create a color map based on node states
        colors = {
            NodeState.SUSCEPTIBLE: 'blue',
            NodeState.INFECTED: 'red', 
            NodeState.REMOVED: 'gray'
        }
        
        node_colors = [colors[self.nodes[n]['state']] for n in self.nodes()]
        
        # Get edge weights for visualization
        edge_weights = [self[u][v]['weight'] for u, v in self.edges()]
        
        # Create the visualization
        pos = nx.spring_layout(self)
        plt.figure(figsize=(10, 8))
        
        nx.draw_networkx_nodes(self, pos, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(self, pos, width=edge_weights, alpha=0.5)
        nx.draw_networkx_labels(self, pos)
        
        plt.title("Information Network State")
        plt.axis('off')
        plt.show()


class NetworkGenerator:
    """Class for generating synthetic networks for testing."""
    
    @staticmethod
    def erdos_renyi(n: int, p: float) -> InformationNetwork:
        """Generate an Erdos-Renyi random graph."""
        network = InformationNetwork()
        graph = nx.erdos_renyi_graph(n, p, directed=True)
        
        # Add nodes with state attribute
        for i in range(n):
            network.add_node(i, state=NodeState.SUSCEPTIBLE)
        
        # Add weighted edges
        for u, v in graph.edges():
            # Random weight between 0 and 1
            weight = np.random.random()
            network.add_edge(u, v, weight=weight)
            
        return network
    
    @staticmethod
    def scale_free(n: int, m: int) -> InformationNetwork:
        """Generate a scale-free network using preferential attachment."""
        network = InformationNetwork()
        graph = nx.barabasi_albert_graph(n, m)
        
        # Add nodes with state attribute
        for i in range(n):
            network.add_node(i, state=NodeState.SUSCEPTIBLE)
        
        # Convert to directed graph by choosing random directions
        for u, v in graph.edges():
            # Choose direction randomly
            if np.random.random() > 0.5:
                # Random weight between 0 and 1
                weight = np.random.random()
                network.add_edge(u, v, weight=weight)
            else:
                weight = np.random.random()
                network.add_edge(v, u, weight=weight)
            
        return network
    
    @staticmethod
    def small_world(n: int, k: int, p: float) -> InformationNetwork:
        """Generate a small-world network."""
        network = InformationNetwork()
        graph = nx.watts_strogatz_graph(n, k, p)
        
        # Add nodes with state attribute
        for i in range(n):
            network.add_node(i, state=NodeState.SUSCEPTIBLE)
        
        # Convert to directed graph by choosing random directions
        for u, v in graph.edges():
            # Choose direction randomly
            if np.random.random() > 0.5:
                weight = np.random.random()
                network.add_edge(u, v, weight=weight)
            else:
                weight = np.random.random()
                network.add_edge(v, u, weight=weight)
            
        return network


class InformationSpreadEnv(gym.Env):
    """Gym environment for the information spread problem."""
    
    def __init__(self, network: InformationNetwork, infection_prob: float = 0.5, 
                 max_steps: int = 100, subnetwork_size_limit: float = 0.02):
        super().__init__()
        
        self.network = network
        self.infection_prob = infection_prob
        self.max_steps = max_steps
        self.current_step = 0
        self.subnetwork_size_limit = subnetwork_size_limit  # Max percentage of nodes in subnetwork
        
        # Define action spaces for both agents
        self.num_nodes = len(network.nodes())
        
        # Agent A action space: select a node to infect
        self.action_space_A = spaces.Discrete(self.num_nodes)
        
        # Agent B action space: choose to remove a node OR reduce weights in a subnetwork
        # Will need custom handling for the subnetwork selection
        self.action_space_B = spaces.Dict({
            'action_type': spaces.Discrete(2),  # 0: remove node, 1: reduce weights
            'node_id': spaces.Discrete(self.num_nodes),
            # For subnetwork selection, custom logic will be implemented in step()
        })
        
        # Observation space will depend on your feature extraction
        # This is a placeholder and will need to be updated
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(low=0, high=1, shape=(self.num_nodes, 5)),  # 5 features per node
            'network_features': spaces.Box(low=0, high=1, shape=(10,)),  # 10 global features
        })
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset all nodes to susceptible
        for node_id in self.network.nodes():
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE
        
        self.current_step = 0
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action_A, action_B):
        """Execute one step in the environment."""
        # First execute Agent A's action (infection)
        self._execute_action_A(action_A)
        
        # Then execute Agent B's action (prevention)
        self._execute_action_B(action_B)
        
        # Natural spread of information through the network
        self._propagate_information()
        
        # Update step counter
        self.current_step += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate rewards
        reward_A = self._calculate_reward_A()
        reward_B = self._calculate_reward_B()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'infected_count': self._get_infected_count(),
            'removed_count': self._get_removed_count(),
        }
        
        return observation, (reward_A, reward_B), done, info
    
    def _execute_action_A(self, action):
        """Execute Agent A's action (infection)."""
        # To be implemented: attempt to infect the selected node
        pass
    
    def _execute_action_B(self, action):
        """Execute Agent B's action (prevention)."""
        # To be implemented: remove node OR reduce weights in subnetwork
        pass
    
    def _propagate_information(self):
        """Propagate information through the network."""
        # To be implemented: SIR model propagation logic
        pass
    
    def _get_observation(self):
        """Generate observation based on current network state."""
        # To be implemented: extract features for RL state
        pass
    
    def _calculate_reward_A(self):
        """Calculate reward for Agent A."""
        # To be implemented: reward based on infection spread
        pass
    
    def _calculate_reward_B(self):
        """Calculate reward for Agent B."""
        # To be implemented: reward based on infection containment
        pass
    
    def _is_done(self):
        """Check if episode is done."""
        # Episode ends if:
        # 1. Max steps reached
        # 2. No susceptible nodes left
        # 3. No infected nodes left (infection contained)
        
        if self.current_step >= self.max_steps:
            return True
        
        susceptible_count = sum(1 for _, data in self.network.nodes(data=True) 
                               if data['state'] == NodeState.SUSCEPTIBLE)
        infected_count = sum(1 for _, data in self.network.nodes(data=True) 
                            if data['state'] == NodeState.INFECTED)
        
        if susceptible_count == 0 or infected_count == 0:
            return True
        
        return False
    
    def _get_infected_count(self):
        """Get the number of infected nodes."""
        return sum(1 for _, data in self.network.nodes(data=True) 
                  if data['state'] == NodeState.INFECTED)
    
    def _get_removed_count(self):
        """Get the number of removed nodes."""
        return sum(1 for _, data in self.network.nodes(data=True) 
                  if data['state'] == NodeState.REMOVED)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            self.network.visualize()
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")


# Playground for testing
def playground():
    """Test environment with synthetic networks."""
    # Generate a small Erdos-Renyi network
    n_nodes = 20
    network = NetworkGenerator.erdos_renyi(n_nodes, 0.2)
    
    # Initialize environment
    env = InformationSpreadEnv(network)
    
    # Reset environment
    obs = env.reset()
    
    # Visualize initial state
    env.render()
    
    # Randomly select initial infection
    initial_infection = np.random.randint(0, n_nodes)
    network.nodes[initial_infection]['state'] = NodeState.INFECTED
    
    # Visualize after infection
    env.render()
    
    print(f"Network has {len(network.nodes())} nodes and {len(network.edges())} edges")
    print(f"Initially infected node: {initial_infection}")


if __name__ == "__main__":
    playground()