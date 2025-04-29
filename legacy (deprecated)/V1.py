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

@dataclass(frozen=True)
class NodeFeature:
    """
        A node feature definition with auto-indexing 
    """
    name: str
    description: str = ""

    _extractor: ExtractorFunc = None
    _index: int = field(init=False, repr=True)
    
    def __post_init__(self):
        # Even though the class is frozen, we can modify via object.__setattr__
        feature_index = FeatureRegistry.register(self.name, self._extractor)
        object.__setattr__(self, '_index', feature_index)
    
    @property
    def index(self) -> int:
        return self._index
    
    # We may need to access specific features/ 
    @property
    def extractor(self) -> ExtractorFunc:
        return self._extractor
    


@dataclass(frozen=True)
class NodeFeatureVector:
    """
        Immutable collection of feature values for a node
    """
    values: Dict[str, float]
    _array: np.darray
    
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
        arr = np.zeros(FeatureRegistry.get_feature_count())
        for name, value in self.values.items():
            arr[FeatureRegistry.get_index(name)] = value
        return arr
    
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
# def _get_infection_pressure(nx: InformationNetwork) -> Dict[int,int]: 
#     """
#         Node feature
#         Calculate infection pressure on susceptible nodes 
#         (weighted sum of number of incoming infected neighbors) 

#         Returns: 
#             Dict[node_id,infection_pressure]

#         TODO: Should this 'pressure' be normalized? 
#     """
#     infection_pressure = {node: 0.0 for node in nx.nodes} 
#     for node in nx.nodes:
#         if nx.nodes[node]['state'] == NodeState.SUSCEPTIBLE: 
#             weighted_pressure = sum(nx[pred][node]['weight']
#                                     for pred in nx.predecessors(node)
#                                     if nx.nodes[pred]['state'] == NodeState.INFECTED) 
#             infection_pressure[node] = weighted_pressure 
#     return infection_pressure

# def _get_spreading_potential(nx: InformationNetwork) -> Dict[int,int]: 
#     """
#         Node feature
#         Identify nodes that are most likely to spread infection 
#         (infected nodes with many susceptible neighbors) 

#         Returns: 
#             Dict[node_id,spreading_potential]
#     """

#     spreading_potential = {node: 0.0 for node in nx.nodes} 
#     infected_nodes = [n for n, d in nx.nodes(data=True) if d['state'] == NodeState.INFECTED]
#     for node in infected_nodes: 
#         weighted_potential = sum(nx[node][succ]['weight']
#                                 for succ in nx.successors(node)
#                                 if nx.nodes[succ]['state'] == NodeState.SUSCEPTIBLE)
#         spreading_potential[node] = weighted_potential
    
#     return spreading_potential



# # NOTE: None of the following functions are used at the moment. 
# def _get_node_features_by_id(nx: InformationNetwork, node_id: int): 
#     """
#         Return features for a single node. 
#         NB: I am not sure we will ever use this, since the most useful 
#         features are probably those that come from the network as a whole. 

#         TODO Implement
#     """
#     pass 

# def _get_node_in_degree(nx: InformationNetwork, node_id, from_node_state, weighted = True): 
#     """
#     Calculate the weighted in-degree proportion from neighbors of given state. 
#     If state is None, returns usual in-degree.
#     Args: 
#         node_id: The ID of the node 
#         node_state: NodeState (see NodeState Enum)
#         weighted: Whether or not in-degree are weighted by edge weights

#     Returns:
#         (Weighted) number of in-edges (from nodes in node_state).
#     """
#     if node_id not in nx: 
#         raise ValueError(f"Node with ID {node_id} not in network.")  

#     # No state given for predecessors, return usual nx.in_degree
#     if from_node_state is None or from_node_state not in NodeState:
#         return nx.in_degree(node_id, weight='weight' ) if weighted is True else nx.in_degree(node_id)
    
#     # Initialize (weighted) degree and regular in-degree
#     state_in_degree = 0

#     # Iterate through all predecessors
#     for pred in nx.predecessors(node_id): 
#         # Check if infected
#         if nx.nodes[pred]['state'] == from_node_state: 
#             state_in_degree += nx[pred][node_id]['weight'] if weighted is True else 1

#     return state_in_degree

# %%
class FeatureExtractors(): 
    """
        This is the interface between our Network and the model agents/training 
        How this works: 
            1 - Create a (private) static method for each feature we want 
            2 - 
    """

    @staticmethod
    def get_state(nx: InformationNetwork) -> Dict[int,float]: 
        """
            States of all nodes in the network, according to NodeState class. 
            Returns: 
                Dict[index_id,normalized_state_value]
        """

        # States Dictionary 
        return {node: nx.nodes[node]['state'].value for node in nx.nodes} 


    @staticmethod 
    def get_internal_influence(nx: InformationNetwork) -> Dict[int,float]:
        """
            Internal influence of infected nodes within the infected subnetwork. 

            Returns: 
                Dict[node_id, internal_influence] 
        """
        infected_nodes = [n for n, d in nx.nodes(data=True) if d['state'] == NodeState.INFECTED]
        
        # Feature 1: Internal influence -- how influential an infected node is 
        # within the subnetwork of infected nodes. 
        internal_influence = {node: 0.0 for node in nx.nodes} 

        # Only create and analyze the subgraph if there are infected nodes
        if infected_nodes:
            infected_subgraph = nx.subgraph(infected_nodes)
            
            # Calculate centrality within the infection subgraph
            # (which infected nodes are central to the infection cluster)
            if len(infected_nodes) > 1:
                try:
                    # Calculate eigenvector centrality in the infected subgraph
                    subgraph_influence = nx.eigenvector_centrality_numpy(infected_subgraph)
                    internal_influence.update(subgraph_influence)

                except nx.AmbiguousSolution:
                    # Fallback to degree centrality if eigenvector fails (e.g., disconnected graph)
                    subgraph_influence = nx.degree_centrality(infected_subgraph)
                    internal_influence.update(subgraph_influence)
                
                except Exception as e: 
                    print(f"Error: {e} (args: {e.args})")
                    subgraph_influence = nx.degree_centrality(infected_subgraph)
                internal_influence.update(subgraph_influence)  
        return internal_influence
    
    @staticmethod 
    def get_infection_pressure(nx: InformationNetwork) -> Dict[int,int]: 
        """
        Node feature
        Calculate infection pressure on susceptible nodes 
        (weighted sum of number of incoming infected neighbors) 

        Returns: 
            Dict[node_id,infection_pressure]

        TODO: Should this 'pressure' be normalized? 
        """
        infection_pressure = {node: 0.0 for node in nx.nodes} 
        for node in nx.nodes:
            if nx.nodes[node]['state'] == NodeState.SUSCEPTIBLE: 
                weighted_pressure = sum(nx[pred][node]['weight']
                                        for pred in nx.predecessors(node)
                                        if nx.nodes[pred]['state'] == NodeState.INFECTED) 
                infection_pressure[node] = weighted_pressure       
        return infection_pressure
    
    @staticmethod
    def get_spreading_potential(nx: InformationNetwork) -> Dict[int,int]: 
        """
            Node feature
            Identify nodes that are most likely to spread infection 
            (infected nodes with many susceptible neighbors) 

            Returns: 
                Dict[node_id,spreading_potential]
        """

        spreading_potential = {node: 0.0 for node in nx.nodes} 
        infected_nodes = [n for n, d in nx.nodes(data=True) if d['state'] == NodeState.INFECTED]
        for node in infected_nodes: 
            weighted_potential = sum(nx[node][succ]['weight']
                                    for succ in nx.successors(node)
                                    if nx.nodes[succ]['state'] == NodeState.SUSCEPTIBLE)
            spreading_potential[node] = weighted_potential
        
        return spreading_potential


# %%
    # NOTE: None of the following functions are used at the moment. 
    # def _get_node_features_by_id(nx: InformationNetwork, node_id: int): 
    #     """
    #         Return features for a single node. 
    #         NB: I am not sure we will ever use this, since the most useful 
    #         features are probably those that come from the network as a whole. 

    #         TODO Implement
    #     """
    #     pass 

    # def _get_node_in_degree(nx: InformationNetwork, node_id, from_node_state, weighted = True): 
    #     """
    #     Calculate the weighted in-degree proportion from neighbors of given state. 
    #     If state is None, returns usual in-degree.
    #     Args: 
    #         node_id: The ID of the node 
    #         node_state: NodeState (see NodeState Enum)
    #         weighted: Whether or not in-degree are weighted by edge weights

    #     Returns:
    #         (Weighted) number of in-edges (from nodes in node_state).
    #     """
    #     if node_id not in nx: 
    #         raise ValueError(f"Node with ID {node_id} not in network.")  

    #     # No state given for predecessors, return usual nx.in_degree
    #     if from_node_state is None or from_node_state not in NodeState:
    #         return nx.in_degree(node_id, weight='weight' ) if weighted is True else nx.in_degree(node_id)
        
    #     # Initialize (weighted) degree and regular in-degree
    #     state_in_degree = 0

    #     # Iterate through all predecessors
    #     for pred in nx.predecessors(node_id): 
    #         # Check if infected
    #         if nx.nodes[pred]['state'] == from_node_state: 
    #             state_in_degree += nx[pred][node_id]['weight'] if weighted is True else 1

    #     return state_in_degree

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
                                    extractors=FeatureExtractors.get_infection_pressure)
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
        
        # self.subnetwork_size_limit = subnetwork_size_limit  # Max percentage of nodes in subnetwork
        
        # Define action spaces for both agents
        # Spreader (Agent A) action space: select a node to infect
        self.action_space_A = spaces.Discrete(self.num_nodes)
        
        # Healer (Agent B) action space: select a node to heal  
        self.action_space_B = spaces.Discrete(self.num_nodes)

        # Define the features and shapes of observable state space.
        # TODO Integrate network features once Agent B has more than one possible action. 
        # 'network_features': spaces.Box(low=0, high=1, shape=(10,)),  # 10 global features
        self.observation_space = spaces.Dict({
            'node_features': 
                spaces.Box(low=0, high=1, shape=(self.num_nodes, FeatureRegistry.get_feature_count())), 
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

        return {
            'node_features': FeatureRegistry.extract_feature_matrix(self.network)
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


