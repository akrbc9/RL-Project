# %% [markdown]
# # 0 - Setup & Class Definitions 

# %%
# Model 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Utilities 
from typing import Dict, List, Tuple, Optional, Set, Union, ClassVar, Any, Type, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from random import seed

# RL 
import gymnasium as gym
from gymnasium import spaces

# %% [markdown]
# ## (i) NetworkX Extension

# %%
class NodeState(Enum):
    SUSCEPTIBLE = 0 
    INFECTED = 1

## NetworkX Extension 
class InformationNetwork(nx.DiGraph):
    """
        Extended directed graph for information spread modeling.
        We extend the model to to add states to nodes and weight to edges automatically. 
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_node(self, node_for_adding, **attr):
        """Add a node with SIR state attribute."""
        if 'state' not in attr:
            attr['state'] = NodeState.SUSCEPTIBLE
            #TODO: Implement other features: malleability, skepticism
        super().add_node(node_for_adding, **attr)
    
    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge with weight attribute."""
        if 'weight' not in attr:
            attr['weight'] = 1.0 # Default weight: 1 
        super().add_edge(u_of_edge, v_of_edge, **attr)

    # Visualization 
    # TODO: Add Graphs as well 
    def visualize(self):
        """
            Visualize the current state of the network.
        """
        # Create a color map based on node states
        colors = {
            NodeState.SUSCEPTIBLE: 'blue',
            NodeState.INFECTED: 'red', 
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
    


# %%

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




# %% [markdown]
# ## (ii) Node/Network State Features Handling

# %%
# Type definition for the extractor function
NetworkType = TypeVar('NetworkType') # Generic type for network nodes
NodeID = TypeVar('NodeID')  # Type for node IDs (typically int or str)
ExtractorFunc = Callable[[NetworkType], Dict[NodeID, float]] 

# Feature Registry and feature pipeline 
class FeatureRegistry:
    """
        Registry that automatically assigns indices to features at runtime
    """
    _feature_indices: Dict[str, int] = {}
    _feature_extractors: Dict[str, int] = {}
    _next_index: int = 0
    
    @classmethod
    def reset(cls):
        """
            Reset the registry
        """
        cls._feature_indices = {}
        cls._feature_extractors = {}
        cls._next_index = 0
    
    @classmethod
    def register(cls, feature_name: str, extractor: Optional[ExtractorFunc]) -> int:
        """
            Register a feature and get its index
            TODO: Add an extractor. 
        """
        if feature_name not in cls._feature_indices:
            cls._feature_indices[feature_name] = cls._next_index
            cls._next_index += 1
        
        if extractor is not None:
            cls._feature_extractors[feature_name] = extractor
            
        return cls._feature_indices[feature_name]
    
    @classmethod
    def get_extractor(cls, feature_name: str) -> ExtractorFunc:
        """Get a feature's extractor function"""
        if feature_name not in cls._feature_extractors:
            raise ValueError(f"No extractor for feature '{feature_name}'")
        return cls._feature_extractors[feature_name]
    
    @classmethod
    def extract_feature(cls, feature_name: str, network: NetworkType) -> Dict[int, float]:
        """
            Extract a feature value from a network

            Returns: 
                A Dictionary of feature values indexed by node_id
        """
        if feature_name not in cls._feature_extractors: 
            raise ValueError(f"No extractor for feature '{feature_name}'")
        extractor = cls.get_extractor(feature_name)
        return extractor(network)
    
    @classmethod
    def extract_all_features(cls, network: NetworkType) -> Dict[str, Dict]:
        """
            Extract all feature values from a network.
            Return: 
                Dictionary of Feature
        """
        return {
            name: cls.extract_feature(name, network) 
            for name in cls._feature_extractors.keys()
        }

    @classmethod
    def extract_feature_matrix(cls, network, **kwargs) -> Tuple[np.ndarray, List[NodeID]]:
        """
        Create a feature matrix with consistent node ordering
        
        Returns:
            Tuple containing:
            - Feature matrix of shape (num_nodes, num_features)
            - List of node IDs in the order they appear in the matrix
        """
        # Extract all features
        all_features = cls.extract_all_features(network, **kwargs)
        
        # Get all unique node IDs from first feature (assuming all features cover same nodes)
        node_ids = network.nodes.keys()

        # Sort node IDs for consistent ordering across calls
        # TODO: If there is a mysterious bug, this is a good place to look. 
        node_ids = list(network.nodes())
        
        # Create feature matrix
        num_nodes = len(node_ids)
        num_features = len(cls._feature_indices)
        feature_matrix = np.zeros((num_nodes, num_features))
        
        # Fill feature matrix
        for feature_name, feature_dict in all_features.items():
            feature_idx = cls._feature_indices[feature_name]
            for i, node_id in enumerate(node_ids):

                # Return default value 0.0 if for whatever reason key is gone. 
                feature_matrix[i, feature_idx] = feature_dict.get(node_id, 0.0)
        
        return feature_matrix, node_ids
    
    @classmethod
    def get_index(cls, feature_name: str) -> int:
        """
            Get a feature's index
            Args: 
                feature_name: str, name of the feature. 
            Return:
                Index of feature in registry. 
        """
        if feature_name not in cls._feature_indices:
            raise ValueError(f"Feature '{feature_name}' not registered")
        return cls._feature_indices[feature_name]
    
    @classmethod
    def get_all_indices(cls) -> Dict[str, int]:
        """
            Get all registered features and their indices
        """
        return cls._feature_indices.copy()
    
    @classmethod
    def get_feature_count(cls) -> int:
        """
            Get the number of registered features
        """
        return len(cls._feature_indices)


# %%

@dataclass #TODO: Frozen = True
class NodeFeature:
    """
        A node feature definition with auto-indexing 
    """
    name: str
    description: str = ""
    extractor: ExtractorFunc = None
    
    _index: int = field(default=None, init=False, repr=True)
    
    def __post_init__(self):
        # Even though the class is frozen, we can modify via object.__setattr__
        feature_index = FeatureRegistry.register(self.name, self.extractor)
        print("Setting Attribute")
        object.__setattr__(self, '_index', feature_index)
    
    @property
    def index(self) -> int:
        return self._index
    
    # We may need to access specific features/ 
    # @property
    # def extractor(self) -> ExtractorFunc:
    #     return self._extractor
    


# @dataclass(frozen=True)
class NodeFeatureVector:
    """
        Immutable collection of feature values for a node
        NOTE: Currently not in use. 
    """
    values: Dict[str, float]
    _array: np.ndarray
    
    def __post_init__(self): 
        # Same trick to modify a frozen class. 
        # Get values with indices in-place. 
        arr = np.zeros(FeatureRegistry.get_feature_count())
        for name, value in self.values.items():
            arr[FeatureRegistry.get_index(name)] = value
        
        object.__setattr__(self,'_array', arr)

    def __getattr__(self, name: str) -> float:
        """Allow accessing features as attributes"""
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"No feature named '{name}'")
    
    def __getitem__(self, idx):
        return self.as_array()[idx] 

    def as_array(self) -> np.ndarray:
        """Convert to a numpy array"""
        return self._array
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'NodeFeatureVector':
        """Create from numpy array"""
        indices = FeatureRegistry.get_all_indices()
        values = {}
        for name, index in indices.items():
            if index < len(arr):
                values[name] = float(arr[index])
        return cls(values)
    
    @classmethod
    def create(cls, **kwargs) -> 'NodeFeatureVector':
        """
            Create with named parameters
            NOTE: Unused 
        """
        return cls(values=kwargs)


# %% [markdown]
# ## (iii) Feature Definitions

# %%
class FeatureExtractors(): 
    """
        This is the interface between our Network and the model agents/training 
        How this works: 
            1 - Create a (private) static method for each feature we want 
            2 - 
    """

    @staticmethod
    def get_state(network: InformationNetwork) -> Dict[int,float]: 
        """
            States of all nodes in the network, according to NodeState class. 
            Returns: 
                Dict[index_id,normalized_state_value]
        """

        # States Dictionary 
        return {node: network.nodes[node]['state'].value for node in network.nodes} 


    @staticmethod 
    def get_internal_influence(network: InformationNetwork) -> Dict[int,float]:
        """
            Internal influence of infected nodes within the infected subnetwork. 

            Returns: 
                Dict[node_id, internal_influence] 
        """
        infected_nodes = [n for n, d in network.nodes(data=True) if d['state'] == NodeState.INFECTED]
        
        # Feature 1: Internal influence -- how influential an infected node is 
        # within the subnetwork of infected nodes. 
        internal_influence = {node: 0.0 for node in network.nodes} 

        if len(infected_nodes) > 2:  # Need at least 3 nodes for meaningful eigenvector centrality
            infected_subgraph = network.subgraph(infected_nodes)
            
            try:
                # Try using NetworkX's eigenvector_centrality with full matrix
                subgraph_influence = nx.eigenvector_centrality(
                    infected_subgraph,
                    max_iter=1000,
                    tol=1e-06,
                    weight='weight'
                )
                internal_influence.update(subgraph_influence)
            except Exception as e:
                # Use degree centrality as fallback
                print(f"Falling back to degree centrality: {str(e)}")
                subgraph_influence = nx.degree_centrality(infected_subgraph)
                internal_influence.update(subgraph_influence)

        return internal_influence
    
    @staticmethod 
    def get_infection_pressure(network: InformationNetwork) -> Dict[int,int]: 
        """
        Node feature
        Calculate infection pressure on susceptible nodes 
        (weighted sum of number of incoming infected neighbors) 

        Returns: 
            Dict[node_id,infection_pressure]

        TODO: Should this 'pressure' be normalized? -> Yes. 
        """
        infection_pressure = {node: 0.0 for node in network.nodes} 
        for node in network.nodes:
            in_degree = network.in_degree(node, weight='weight')
            if network.nodes[node]['state'] == NodeState.SUSCEPTIBLE and in_degree > 0: 
                weighted_pressure = sum(network[pred][node]['weight']
                                        for pred in network.predecessors(node)
                                        if network.nodes[pred]['state'] == NodeState.INFECTED) / in_degree
                infection_pressure[node] = weighted_pressure       
        return infection_pressure
    
    @staticmethod
    def get_spreading_potential(network: InformationNetwork) -> Dict[int,int]: 
        """
            Node feature
            Identify nodes that are most likely to spread infection 
            (infected nodes with many susceptible neighbors) 

            Returns: 
                Dict[node_id,spreading_potential]
        """

        spreading_potential = {node: 0.0 for node in network.nodes} 
        infected_nodes = [n for n, d in network.nodes(data=True) if d['state'] == NodeState.INFECTED]
        for node in infected_nodes: 
            out_degree = network.out_degree(node, weight='weight')
            if out_degree > 0:
                weighted_potential = sum(network[node][succ]['weight']
                                        for succ in network.successors(node)
                                        if network.nodes[succ]['state'] == NodeState.SUSCEPTIBLE)
                spreading_potential[node] = weighted_potential / out_degree
        
        return spreading_potential


# %%
# NOTE: This is where features are registered for the Gym Environment.
@dataclass
class NodeFeatures:
    """
        Collection of feature definitions
    """
    # Reset registry when class is loaded
    FeatureRegistry.reset()
    
    # Features will be automatically indexed in the order defined
    STATE = NodeFeature(name="state", description="Node infection state",
                        extractor=FeatureExtractors.get_state)
    INTERNAL_INFLUENCE = NodeFeature(name="internal_influence",
                        extractor=FeatureExtractors.get_internal_influence)
    INFECTION_PRESSURE = NodeFeature(name="infection_pressure",
                                    extractor=FeatureExtractors.get_infection_pressure)
    SPREADING_POTENTIAL = NodeFeature(name="spreading_potential",
                                    extractor=FeatureExtractors.get_spreading_potential)
    
    @classmethod
    def get_all(cls) -> List[NodeFeature]:
        """
            Get all feature definitions
        """
        return [
            cls.STATE, 
            cls.INTERNAL_INFLUENCE,
            cls.INFECTION_PRESSURE,
            cls.SPREADING_POTENTIAL
        ]

# %% [markdown]
# ## (iv) Gym Environment (RL)

# %%
class InformationSpreadEnv(gym.Env):
    """
        Gym environment for the information spread problem.
        TODO: Decouple model from environment container. 
    """
    
    def __init__(self, network: InformationNetwork, infection_prob: float = 0.5, 
                 recovery_prob: float = 0.05, max_steps: int = 100,
                 subnetwork_size_limit: float = 0.02):
        super().__init__()
        
        # Underlying (Social) Network 
        self.network = network

        # Model Parameters  
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob

        # Simulation Parameters 
        self.max_steps = max_steps
        self.current_step = 0
        self.num_nodes = len(network.nodes())

        # Simulation Data 
        self.previous_infection_rate = 0 
        self.node_ids = []
    
        # self.subnetwork_size_limit = subnetwork_size_limit  # Max percentage of nodes in subnetwork
        
        # Define action spaces for both agents
        # Spreader (Agent A) action space: select a node to infect
        self.action_space_A = spaces.Discrete(self.num_nodes)
        
        # Healer (Agent B) action space: select a node to heal  
        self.action_space_B = spaces.Discrete(self.num_nodes)

        # Define the features and shapes of observable state space.
        # TODO Integrate network features once Agent B has more than one possible action. 
        # 'network_features': spaces.Box(low=-1, high=1, shape=(10,)),  # 10 global features
        self.observation_space = spaces.Dict({
            'node_features': 
                spaces.Box(low=-1, high=1, shape=(self.num_nodes, FeatureRegistry.get_feature_count())), 
        })
    
    def reset(self):
        # Reset all nodes to susceptible
        for node_id in self.network.nodes():
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE
        
        self.current_step = 0
        
        # Return initial observation and info
        return self._get_observation(), {}
    
    def step(self, action_A, action_B):
        """
            Execute one step in the environment.
            TODO: Consider adding delayed model update instructions 
            to save on compute time. 
        """
        # First execute Agent A's action (infection)
        self._execute_action_A(action_A)
        
        # Then execute Agent B's action (prevention)
        self._execute_action_B(action_B)
        
        # Natural spread of information through the network
        self._propagate_information()
        
        # Natural recovery through neighbor influence 
        self._process_recovery() 

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
        # TODO TODO TODO : Why do we use info? 
        info = {
            'infected_count': self._get_infected_count(),
        }
        
        return observation, (reward_A, reward_B), done, info
    
    # This is the logic for Spreader Agent (Agent A)'s
    def _execute_action_A(self, action):
        """
            Execute Agent A's action (infection).
            Args: 
                action: ID of node to infect.
        """
        node_id = action 

        if node_id not in self.network: 
            raise ValueError(f"Invalid node {node_id} passed to Agent A (Spreader")
        
        # If node is susceptible and infection threshold is reached:
        # TODO: Include malleability, skepticism. 
        if (self.network.nodes[node_id]['state'] == NodeState.SUSCEPTIBLE
            and np.random.random() < self.infection_prob):
            self.network.nodes[node_id]['state'] = NodeState.INFECTED

    # This is the logic for Spreader Agent (Agent A)'s
    def _execute_action_B(self, action):
        """
            Exectute Agent B's action (recovery)
            Args: 
                action: ID of node to try and recover
        """
        node_id = action 

        if node_id not in self.network: 
            raise ValueError(f"Invalid node {node_id} passed to Agent B (Healer)")
        
        # If node is infected and recovery threshold is reached: 
        # TODO: Include malleability, skepticism. 
        # TODO: If node goes from INFECTED --> SUSCEPTIBLE, add cooldown period where they can't get infected. 
        if (self.network.nodes[node_id]['state'] == NodeState.INFECTED
            and np.random.random() < self.recovery_prob):
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE


    def _propagate_information(self):
        """
            Propagate information through the network using SIR model.
            Returns: 
                (int,int): number of new infections and number of recoveries. 
        """
        # Track nodes that will be newly infected this step
        new_infections = []

        # Infection process
        # For each infected node
        for node_id in self.network.nodes:
            if self.network.nodes[node_id]['state'] != NodeState.INFECTED:
                continue
            
            # Attempt to infect each susceptible neighbor
            for neighbor_id in self.network.successors(node_id):
                if self.network.nodes[neighbor_id]['state'] != NodeState.SUSCEPTIBLE:
                    continue
                
                # Use edge weight to influence probability
                edge_weight = self.network[node_id][neighbor_id]['weight']
                infection_prob = self.infection_prob * edge_weight
                
                # Probabilistic infection attempt
                if np.random.random() < infection_prob:
                    new_infections.append(neighbor_id)
        
        # Apply new infections
        for node_id in new_infections:
            self.network.nodes[node_id]['state'] = NodeState.INFECTED
        
        return len(new_infections)

    def _process_recovery(self):
        """
        Propagate recovery through neighboring edges 
        TODO Implement the network recovery logic
        """
        recoveries = []
        # Recovery process
        # For each infected node, there's a chance to recover
        for node_id in self.network.nodes:
            if self.network.nodes[node_id]['state'] == NodeState.INFECTED:
                if np.random.random() < self.recovery_prob:  # You'd need to add this parameter
                    recoveries.append(node_id)
         
        for node_id in recoveries: 
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE
        
        return len(recoveries) 
    
    def _get_observation(self):
        """
            Generate observation based on current network state.
        """
        feature_matrix, node_ids = FeatureRegistry.extract_feature_matrix(self.network)
        self.node_ids = node_ids
        return {
            'node_features': feature_matrix
        }

    def _calculate_reward_A(self):
        current_infection_rate = self._get_infected_count() / self.num_nodes
        infection_change = current_infection_rate - self.previous_infection_rate
        self.previous_infection_rate = current_infection_rate
        
        # Base reward on infection growth
        growth_reward = infection_change * 20.0  # Scaling factor
        
        # Add small reward for maintaining high infection
        persistence_reward = current_infection_rate * 2.0
        
        # Penalty for having infections clustered (vulnerable to containment)
        clustering_penalty = 0
        if self._get_infected_count() > 3:  # Only if enough infected nodes
            infected_subgraph = self.network.subgraph([n for n, d in self.network.nodes(data=True) 
                                                if d['state'] == NodeState.INFECTED])
            if len(infected_subgraph) > 1:
                clustering_penalty = nx.transitivity(infected_subgraph) * 5.0
        
        return growth_reward + persistence_reward - clustering_penalty
    
    def _calculate_reward_B(self):
        current_infection_rate = self._get_infected_count() / self.num_nodes
        infection_change = current_infection_rate - self.previous_infection_rate
        
        # Base reward on infection reduction
        containment_reward = -infection_change * 20.0  # Negative change is good
        
        # Bonus for keeping infection rate low
        low_infection_bonus = (1.0 - current_infection_rate) * 2.0

        return containment_reward + low_infection_bonus
    
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
    
    def _get_susceptible_count(self):
        """Get the number of removed nodes."""
        return sum(1 for _, data in self.network.nodes(data=True) 
                  if data['state'] == NodeState.SUSCEPTIBLE)
    
    def render(self):
        """Render the environment."""
        self.network.visualize()

# %% [markdown]
# # 1 - Playground and Visualizations 

# %%
import IPython.display as display
import time
import ray
from ray import tune 
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

# %%
class MultiAgentNetworkEnv(MultiAgentEnv):
    """
        We define a multi-agent wrapper for the environment. 
        This allows us to e.g. decouple observations between agents down the lines. 
        
        This also follows Ray RLLib's best practices for MARL. 
    """
    def __init__(self, env_config):
        super(MultiAgentNetworkEnv, self).__init__()  # Correct way
        # Initialize with your network environment
        self.env = InformationSpreadEnv(**env_config)
        self.possible_agents = ['agent_a','agent_b']
        self._agent_ids = ['agent_a','agent_b']
        
        # Set up action and observation spaces for agents
        self.action_space = {
            "agent_a": self.env.action_space_A,
            "agent_b": self.env.action_space_B
        }
        self.observation_space = {
            "agent_a": self.env.observation_space,
            "agent_b": self.env.observation_space
        }
    
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset()
        # return observation dict and infos dict.
        return {"agent_a": obs, "agent_b": obs}, {}
    
    def step(self, action_dict):
        # Unpack actions from both agents
        action_A = action_dict["agent_a"]
        action_B = action_dict["agent_b"]
        
        # Execute environment step
        obs, rewards, done, info = self.env.step(action_A, action_B)
        
        # Format for multi-agent environment
        observations = {
            "agent_a": obs,
            "agent_b": obs
        }
        rewards = {
            "agent_a": rewards[0],
            "agent_b": rewards[1]
        }
        dones = {
            "agent_a": done,
            "agent_b": done,
            "__all__": done
        }
        
                # New Gymnasium API requires "truncateds" to indicate time limit cutoffs
        # (false if terminated normally, true if cut off by time limit)
        truncateds = {
            "agent_a": False,
            "agent_b": False,
            "__all__": False
        }
        
        infos = {
            "agent_a": info,
            "agent_b": info
        }

        return observations, rewards, dones, truncateds, infos

    def action_space_sample(self, agent_ids=None):
        """Override to provide proper action space sampling.
        
        Args:
            agent_ids: Optional list of agent IDs to sample for
                    If None, sample for all agents
        
        Returns:
            Dictionary mapping agent IDs to sampled actions
        """
        if agent_ids is None:
            agent_ids = self.possible_agents
            
        return {
            agent_id: self.action_space[agent_id].sample()
            for agent_id in agent_ids
        }

    def observation_space_sample(self, agent_ids=None):
        """Override to provide proper observation space sampling.
        
        Args:
            agent_ids: Optional list of agent IDs to sample for
                    If None, sample for all agents
        
        Returns:
            Dictionary mapping agent IDs to sampled observations
        """
        if agent_ids is None:
            agent_ids = self.possible_agents
            
        return {
            agent_id: self.observation_space[agent_id].sample()
            for agent_id in agent_ids
        }


def env_creator(env_config):
    return MultiAgentNetworkEnv(env_config)

# 1. Initialize Ray
ray.shutdown()
ray.init()

# Register the environment with Ray
register_env("network_spread", env_creator)

# First create the environment instance
test_env = MultiAgentNetworkEnv(env_config={
    "network": NetworkGenerator.erdos_renyi(50, 0.2),
    "infection_prob": 0.5,
    "recovery_prob": 0.1,
    "max_steps": 100
})

# Then use its observation spaces in the config
config = (
    PPOConfig()
    .environment("network_spread", env_config = {          # Change env_config format
        "network": NetworkGenerator.erdos_renyi(50, 0.2),
        "infection_prob": 0.5,
        "recovery_prob": 0.1,
        "max_steps": 100
    })
    .multi_agent(
        policies={
            "agent_a": PolicySpec(
                observation_space=test_env.observation_space["agent_a"],
                action_space=test_env.action_space["agent_a"]
            ),
            "agent_b": PolicySpec(
                observation_space=test_env.observation_space["agent_b"],
                action_space=test_env.action_space["agent_b"]
            )
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
    )
    .framework("torch")
    .training(
        train_batch_size=4000,
        lr=5e-5,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.01,
    )
    .rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=200
    )
    .debugging(log_level="WARNING")
)

# Run training
results = tune.run(
    "PPO",
    config=config.to_dict(),
    stop={
        "training_iteration": 100,
        "episode_reward_mean": 195.0  # Optional: stop when reward threshold reached
    },
    checkpoint_freq=10,
    checkpoint_at_end=True,
    # metrics_export_format=["csv", "tensorboard"],
    local_dir="./ray_results"  # Directory to store results
)

# Get best performing trial
best_trial = results.get_best_trial("episode_reward_mean", "max")
print(f"Best trial final reward: {best_trial.last_result['episode_reward_mean']}")

print(results.results_df)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output, display
import numpy as np

# %%
# Step 1: Find the checkpoint
checkpoint_path = results.get_best_checkpoint(
    trial=results.get_best_trial(metric="episode_reward_mean", mode="max"),
    metric="episode_reward_mean",
    mode="max"
)

# Step 2: Rebuild config (optional if you saved it)
# If you have 'config' still in memory, you can reuse it. Otherwise rebuild same config.

# Step 3: Create trainer
trainer = config.build()

# Step 4: Restore from checkpoint
trainer.restore(checkpoint_path)

# First create the environment instance
test_env = MultiAgentNetworkEnv(env_config={
    "network": NetworkGenerator.erdos_renyi(50, 0.2),
    "infection_prob": 0.5,
    "recovery_prob": 0.1,
    "max_steps": 1000
})

trained_agents_play(test_env, trainer, episodes=20, delay=100)


# %%
def visualize_trained_agents(trained_policies, num_episodes=5, max_steps=100, network_size=50, edge_prob=0.2):
    """
    Visualize trained agents playing against each other in the environment.
    
    Args:
        trained_policies: Dictionary mapping agent IDs to trained policies
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        network_size: Number of nodes in the network
        edge_prob: Probability of edge creation in Erdos-Renyi network
    """
    # Generate a new network for testing
    network = NetworkGenerator.erdos_renyi(network_size, edge_prob)
    
    # Initialize environment
    env_config = {
        "network": network,
        "infection_prob": 0.5,
        "recovery_prob": 0.1,
        "max_steps": max_steps
    }
    env = MultiAgentNetworkEnv(env_config)
    
    # Statistics for multiple episodes
    episode_lengths = []
    episode_infected_counts = []
    episode_rewards_a = []
    episode_rewards_b = []
    
    # Run multiple episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Randomly select initial infection
        initial_infection = np.random.randint(0, network_size)
        env.env.network.nodes[initial_infection]['state'] = NodeState.INFECTED
        
        print(f"Network has {len(env.env.network.nodes())} nodes and {len(env.env.network.edges())} edges")
        print(f"Initially infected node: {initial_infection}")
        
        # Metrics to track for this episode
        step_infected_counts = []
        step_susceptible_counts = []
        total_reward_a = 0
        total_reward_b = 0
        
        # Print initial state
        print(f"Step 0: {env.env._get_infected_count()} infected, {env.env._get_susceptible_count()} susceptible")
        
        # Create figure for real-time plotting
        plt.figure(figsize=(12, 8))
        
        for step in range(1, max_steps + 1):
            # Get actions from trained policies
            action_a = trained_policies["agent_a"].compute_single_action(obs["agent_a"])[0]
            action_b = trained_policies["agent_b"].compute_single_action(obs["agent_b"])[0]
            
            # Execute step
            action_dict = {"agent_a": action_a, "agent_b": action_b}
            obs, rewards, dones, truncateds, infos = env.step(action_dict)
            
            # Track metrics
            infected_count = env.env._get_infected_count()
            susceptible_count = env.env._get_susceptible_count()
            step_infected_counts.append(infected_count)
            step_susceptible_counts.append(susceptible_count)
            total_reward_a += rewards["agent_a"]
            total_reward_b += rewards["agent_b"]
            
            # Print current state every 10 steps
            if step % 10 == 0 or step == 1:
                print(f"Step {step}: {infected_count} infected, {susceptible_count} susceptible")
                print(f"  Rewards - Agent A: {rewards['agent_a']:.2f}, Agent B: {rewards['agent_b']:.2f}")
            
            # Real-time plot update (optional, can be commented out for faster simulation)
            if step % 10 == 0:
                plt.clf()
                steps = range(1, len(step_infected_counts) + 1)
                plt.plot(steps, step_infected_counts, 'r-', label='Infected')
                plt.plot(steps, step_susceptible_counts, 'b-', label='Susceptible')
                plt.xlabel('Step')
                plt.ylabel('Count')
                plt.title(f'Episode {episode+1}: Infection Spread Over Time')
                plt.legend()
                plt.grid(True)
                plt.pause(0.01)
            
            # Break if done
            if dones["__all__"]:
                print(f"\nEpisode ended at step {step}")
                if infected_count == 0:
                    print("Infection eradicated - Healer (Agent B) wins!")
                elif susceptible_count == 0:
                    print("All nodes infected - Spreader (Agent A) wins!")
                else:
                    print("Maximum steps reached")
                break
        
        # Save episode statistics
        episode_lengths.append(step)
        episode_infected_counts.append(step_infected_counts)
        episode_rewards_a.append(total_reward_a)
        episode_rewards_b.append(total_reward_b)
        
        # Final plot for this episode
        plt.figure(figsize=(10, 6))
        steps = range(1, len(step_infected_counts) + 1)
        plt.plot(steps, step_infected_counts, 'r-', label='Infected')
        plt.plot(steps, step_susceptible_counts, 'b-', label='Susceptible')
        plt.xlabel('Step')
        plt.ylabel('Count')
        plt.title(f'Episode {episode+1}: Infection Spread Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'episode_{episode+1}_spread.png')
        plt.close()
        
        # Network visualization (uncomment if you have implemented network.visualize())
        # env.env.render()
    
    # Summarize results
    print("\n--- Summary ---")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
    print(f"Average final infection rate: {np.mean([counts[-1] for counts in episode_infected_counts])/network_size:.2%}")
    print(f"Average Agent A (Spreader) total reward: {np.mean(episode_rewards_a):.2f}")
    print(f"Average Agent B (Healer) total reward: {np.mean(episode_rewards_b):.2f}")
    
    # Plot summary across episodes
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Episode lengths
    plt.subplot(2, 2, 1)
    plt.bar(range(1, num_episodes + 1), episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.grid(True)
    
    # Plot 2: Final infection rates
    plt.subplot(2, 2, 2)
    final_infection_rates = [counts[-1]/network_size for counts in episode_infected_counts]
    plt.bar(range(1, num_episodes + 1), final_infection_rates)
    plt.xlabel('Episode')
    plt.ylabel('Final Infection Rate')
    plt.title('Final Infection Rates')
    plt.grid(True)
    
    # Plot 3: Agent rewards
    plt.subplot(2, 2, 3)
    x = range(1, num_episodes + 1)
    width = 0.35
    plt.bar([i - width/2 for i in x], episode_rewards_a, width, label='Agent A (Spreader)')
    plt.bar([i + width/2 for i in x], episode_rewards_b, width, label='Agent B (Healer)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Average infection progression
    plt.subplot(2, 2, 4)
    # Find the max length and pad shorter episodes with their final values
    max_length = max(len(counts) for counts in episode_infected_counts)
    padded_counts = []
    for counts in episode_infected_counts:
        if len(counts) < max_length:
            padded = counts + [counts[-1]] * (max_length - len(counts))
        else:
            padded = counts
        padded_counts.append(padded)
    
    avg_infected = np.mean(padded_counts, axis=0)
    plt.plot(range(1, max_length + 1), avg_infected, 'r-')
    plt.xlabel('Step')
    plt.ylabel('Average Infected Count')
    plt.title('Average Infection Progression')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('summary_plots.png')
    plt.show()
    
    return {
        'episode_lengths': episode_lengths,
        'infected_counts': episode_infected_counts,
        'rewards_a': episode_rewards_a,
        'rewards_b': episode_rewards_b
    }

def evaluate_trained_agents():
    # Load the best checkpoint from your training
    checkpoint_path = results.get_best_checkpoint(
        trial=results.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
        mode="max"
    )
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create an Algorithm object from the checkpoint
    trained_algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Get the trained policies
    trained_policies = {
        "agent_a": trained_algo.get_policy("agent_a"),
        "agent_b": trained_algo.get_policy("agent_b")
    }
    
    # Run visualization and evaluation
    stats = visualize_trained_agents(
        trained_policies=trained_policies,
        num_episodes=5,
        max_steps=100, 
        network_size=50,
        edge_prob=0.2
    )
    
    return stats

evaluation_stats = evaluate_trained_agents()


# %%



