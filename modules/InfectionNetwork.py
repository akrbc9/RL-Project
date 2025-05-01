# Model 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass, field
from enum import Enum

class NodeState(Enum):
    SUSCEPTIBLE = 0 
    INFECTED = 1

class NetworkType(Enum): 
    ERDOS_RENYI = "erdos_renyi"
    SCALE_FREE  = "scale_free"
    SMALL_WORLD = "small_world"


## NetworkX Extension 
class InformationNetwork(nx.DiGraph):
    """
        Extended directed graph for information spread modeling.
        We extend the model to to add states to nodes and weight to edges automatically. 
    """
    
    def __init__(self, *args, **kwargs):
        self.type = "None"
        super().__init__(*args, **kwargs)
    
    def add_node(self, node_for_adding, **attr):
        """Add a node with SIR state, skepticism, infection length attributes."""
        if 'state' not in attr:
            attr['state'] = NodeState.SUSCEPTIBLE
        if 'skepticism' not in attr: 
            attr['skepticism'] = random.random()
        if 'length_of_infection' not in attr: 
            attr['length_of_infection'] = 0 
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


class NetworkGenerator:
    """
        Class for generating synthetic networks for testing.
    """
    
    @staticmethod
    def erdos_renyi(n: int, p: float) -> InformationNetwork:
        """Generate an Erdos-Renyi random graph."""
        network = InformationNetwork()
        network.type = NetworkType.ERDOS_RENYI
        graph = nx.erdos_renyi_graph(n, p, directed=True)
        
            # Add nodes with state attribute
        for node in graph.nodes():
                network.add_node(node, 
                                state=NodeState.SUSCEPTIBLE, # TODO: Custom init for skepticism (for parameter sweep).
                                skepticism=np.random.beta(3, 7),  # Values between 0-1
                                infection_length=0)
        # Add weighted edges
        for u, v in graph.edges():
            # Random weight between 0 and 1
            weight = np.random.random()
            network.add_edge(u, v, weight=weight)
            
        return network
    
    @staticmethod
    def scale_free(n: int, m: int) -> InformationNetwork:
        """Generate a scale-free network using preferential attachment."""
        # Create empty network
        network = InformationNetwork()
        network.type = NetworkType.SCALE_FREE

        # Create the networkx graph 
        graph = nx.barabasi_albert_graph(n, m)
        

        
        # Add all nodes with COMPLETE initialization
        for node in graph.nodes():
            network.add_node(node, 
                            state=NodeState.SUSCEPTIBLE,
                            skepticism=np.random.beta(3, 7),  # Values between 0-1
                            infection_length=0)
        
        # Add edges with random directions and weights
        for u, v in graph.edges():
            if np.random.random() > 0.5:
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
        network.type = NetworkType.SMALL_WORLD
        graph = nx.watts_strogatz_graph(n, k, p)
        
        for node in graph.nodes():
                network.add_node(node, 
                                state=NodeState.SUSCEPTIBLE,
                                skepticism=np.random.beta(3, 7),  # Values between 0-1
                                infection_length=0)
        
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

