
from typing import Dict, List, Tuple, Optional,Any, Type, Callable, TypeVar
from dataclasses import dataclass, field
from InfectionNetwork import InformationNetwork, NodeState
import numpy as np
import networkx as nx 

# Type definition for the extractor function
NetworkTypeVar = TypeVar('NetworkType') # Generic type for network nodes
NodeID = TypeVar('NodeID')  # Type for node IDs (typically int or str)
ExtractorFunc = Callable[[NetworkTypeVar], Dict[NodeID, float]] 


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
    def extract_feature(cls, feature_name: str, network: NetworkTypeVar) -> Dict[int, float]:
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
    def extract_all_features(cls, network: NetworkTypeVar) -> Dict[str, Dict]:
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


@dataclass #TODO: Frozen = True
class NodeFeature:
    """
        A node feature definition with auto-indexing 
    """
    name: str
    typeription: str = ""
    extractor: ExtractorFunc = None
    
    _index: int = field(default=None, init=False, repr=True)
    
    def __post_init__(self):
        # Even though the class is frozen, we can modify via object.__setattr__
        feature_index = FeatureRegistry.register(self.name, self.extractor)
        # print(f"Registered feature {self.name}")
        object.__setattr__(self, '_index', feature_index)
    
    @property
    def index(self) -> int:
        return self._index

# TODO: Continue Here

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
    def get_skepticism(network: InformationNetwork) -> Dict[int,float]: 
        return {node: network.nodes[node]['skepticism'] for node in network.nodes}

    @staticmethod
    def get_length_of_infection(network: InformationNetwork) -> Dict[int,float]: 
        lengths = {node: network.nodes[node]['length_of_infection']
                    for node in network.nodes}
        max_length = max(lengths.values() if lengths else 1) # Normalization factor  
        if max_length == 0: max_length = 1
        return {node: network.nodes[node]['length_of_infection']/max_length for node in network.nodes}

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
                # For example if sub-graph is disconnected
                # print(f"Falling back to degree centrality: {str(e)}")
                subgraph_influence = nx.degree_centrality(infected_subgraph)
                internal_influence.update(subgraph_influence)

        return internal_influence
    
    @staticmethod 
    def get_infection_pressure(network: InformationNetwork) -> Dict[int,int]: 
        """
        Node feature
        Calculate (normalized) infection pressure on susceptible nodes 
        (weighted sum of number of incoming infected neighbors) 

        Returns: 
            Dict[node_id,infection_pressure]

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

# NOTE: This is where features are registered for the Training Environment.
@dataclass
class NodeFeatures:
    """
        Collection of feature definitions
    """
    # Reset registry when class is loaded
    FeatureRegistry.reset()
    
    # Features will be automatically indexed in the order defined
    STATE = NodeFeature(name="state", typeription="Node infection state",
                        extractor=FeatureExtractors.get_state)
    SKEPTICISM = NodeFeature(name="skepticism", extractor=FeatureExtractors.get_skepticism)
    LENGTH_OF_INFECTION = NodeFeature(name="length_of_infection", extractor=FeatureExtractors.get_length_of_infection)
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
            cls.SKEPTICISM,
            cls.LENGTH_OF_INFECTION,
            cls.INFECTION_PRESSURE,
            cls.SPREADING_POTENTIAL
        ]



