import numpy as np
import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Union
import gymnasium as gym
from gymnasium import spaces


class NodeState(Enum):
    """Enum for the possible states of a node in the SIR model."""
    SUSCEPTIBLE = 0
    INFECTED = 1


class FeatureDimensions(Enum):
    """
        Enum defining dimensions of feature vectors in the state representation
        TODO Make sure this is integrated in all parts of the code. 
    """
    NODE_FEATURE_COUNT = 4   # Number of features per node
    NETWORK_FEATURE_COUNT = 10  # Number of global network features

class FeatureNames(Enum): 
    STATE = 'state'
    INTERNAL_INFLUENCE = 'internal_influence' 
    INFECTION_PRESSURE = 'infection_pressure'
    SPREADING_POTENTIAL = 'spreading_potential'
    NODE_FEATURES = 'node_features'

class InformationNetwork(nx.DiGraph):
    """Extended directed graph for information spread modeling."""
    
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
    
    # def get_node_features(self, node_id): 
    #     if node_id is None or node_id not in self: 
    #         raise ValueError(f"InformationNetwork: Node {node_id} not in network.")
    #     node_data = self.nodes[node_id]

    #     # Gather node features
    #     node_state = node_data['state'].value / 2.0 # Normalize to [0,1]. 
    #     in_degree = self.in_degree(node_id, weight='weight') / len(self)
    #     out_degree= self.out_degree(node_id, weight='weight') / len(self)

    # Visualization 
    # TODO: Add Graphs as well 
    def visualize(self):
        """Visualize the current state of the network."""
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
    
    # Feature Extraction Methods
    # Node features 

    def get_node_in_degree(self, node_id, from_node_state, weighted = True): 
        """
        Calculate the weighted in-degree proportion from neighbors of given state. 
        If state is None, returns usual in-degree.
        Args: 
            node_id: The ID of the node 
            node_state: NodeState (see NodeState Enum)
            weighted: Whether or not in-degree are weighted by edge weights

        Returns:
            (Weighted) number of in-edges (from nodes in node_state).
        """
        if node_id not in self: 
            raise ValueError(f"Node with ID {node_id} not in network.")  

        # No state given for predecessors, return usual nx.in_degree
        if from_node_state is None or from_node_state not in NodeState:
            return self.in_degree(node_id, weight='weight' ) if weighted is True else self.in_degree(node_id)
        
        # Initialize (weighted) degree and regular in-degree
        state_in_degree = 0

        # Iterate through all predecessors
        for pred in self.predecessors(node_id): 
            # Check if infected
            if self.nodes[pred]['state'] == from_node_state: 
                state_in_degree += self[pred][node_id]['weight'] if weighted is True else 1

        return state_in_degree
    
    def get_node_features(self):
        """
        Calculate centrality measures on infected subnetwork
        Return:
            Dictionary of various metrics. 
        
        TODO: Pluralize names
        """
        infected_nodes = [n for n, d in self.nodes(data=True) if d['state'] == NodeState.INFECTED]
        
        # States Dictionary 
        state = {node: self.nodes[node]['state'].value / 2.0 for node in self.nodes} 

        # Feature 1: Internal influence -- how influential an infected node is 
        # within the subnetwork of infected nodes. 
        internal_influence = {node: 0.0 for node in self.nodes} 

        # Only create and analyze the subgraph if there are infected nodes
        if infected_nodes:
            infected_subgraph = self.subgraph(infected_nodes)
            
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
    
        # Calculate infection pressure on susceptible nodes 
        # (weighted sum of infected neighbors) 
        # TODO: Should this 'pressure' be normalized? 
        infection_pressure = {node: 0.0 for node in self.nodes} 
        for node in self.nodes: 
            if self.nodes[node]['state'] == NodeState.SUSCEPTIBLE: 
                weighted_pressure = sum(self[pred][node]['weight']
                                        for pred in self.predecessors(node)
                                        if self.nodes[pred]['state'] == NodeState.INFECTED) 
                infection_pressure[node] = weighted_pressure 
        
        # Identify nodes that are most likely to spread infection 
        # (infected nodes with many susceptible neighbors) 
        spreading_potential = {node: 0.0 for node in self.nodes} 
        for node in infected_nodes: 
            weighted_potential = sum(self[node][succ]['weight']
                                    for succ in self.successors(node)
                                    if self.nodes[succ]['state'] == NodeState.SUSCEPTIBLE)
            spreading_potential[node] = weighted_potential
        
        return { 
            FeatureNames.STATE: state,
            FeatureNames.INTERNAL_INFLUENCE: internal_influence,
            FeatureNames.INFECTION_PRESSURE: infection_pressure, 
            FeatureNames.SPREADING_POTENTIAL: spreading_potential, 
        }

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
                 recovery_prob: float = 0.05, max_steps: int = 100,
                 subnetwork_size_limit: float = 0.02):
        super().__init__()
        
        self.network = network
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.max_steps = max_steps
        self.current_step = 0

        # Historical Data 
        self.previous_infection_rate = 0 
        
        # self.subnetwork_size_limit = subnetwork_size_limit  # Max percentage of nodes in subnetwork
        
        # Define action spaces for both agents
        self.num_nodes = len(network.nodes())
        
        # Spreader (Agent A) action space: select a node to infect
        self.action_space_A = spaces.Discrete(self.num_nodes)
        
        # Healer (Agent B) action space: select a node to heal  
        self.action_space_B = spaces.Discrete(self.num_nodes)

        # Healer action space: choose to remove a node OR reduce weights in a subnetwork
        # Will need custom handling for the subnetwork selection
        # self.action_space_B = spaces.Dict({
        #     'action_type': spaces.Discrete(2),  # 0: remove node, 1: reduce weights
        #     'node_id': spaces.Discrete(self.num_nodes),
        #     # For subnetwork selection, custom logic will be implemented in step()
        # })
        
        # Observation space depends on extracted features.
        # Node features: 
        #   1 - State (SUSCEPTIBLE, INFECTED)
        #   2 - Spreading Potential (0 to 1)
        #   3 - Infection Pressure (0 to 1)
        #   6 - internal influence (0 to 1)
        #
        #   DELAYED: 
        #   4 - in-degree (0 to 1)  TODO: Make weighted
        #   5 - out-degree (0 to 1) TODO: Make weighted
        #
        # Global (network) features:
        #   1 - Proportion Infected Nodes
        #   2 - Clustering measure?
        #   3 - Community structure?
        #   TODO: Add more? 
        self.observation_space = spaces.Dict({
            FeatureNames.NODE_FEATURES: spaces.Box(low=0, high=1, shape=(self.num_nodes, FeatureDimensions.NODE_FEATURE_COUNT.value)), 

            # TODO Integrate network features once Agent B has more than one possible action. 
            # 'network_features': spaces.Box(low=0, high=1, shape=(10,)),  # 10 global features
        })
    
    def reset(self):
        # Reset all nodes to susceptible
        for node_id in self.network.nodes():
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE
        
        self.current_step = 0
        
        # Return initial observation and info
        return self._get_observation(), {}
    
    def step(self, action_A, action_B):
        """Execute one step in the environment."""
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
        info = {
            'infected_count': self._get_infected_count(),
        }
        
        return observation, (reward_A, reward_B), done, info
    
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
        # Node features: 
        #   1 - State (SUSCEPTIBLE, INFECTED)
        #   2 - Spreading Potential (0 to 1)
        #   3 - Infection Pressure (0 to 1)
        #   6 - internal influence (0 to 1)
        #
        #   DELAYED: 
        #   4 - in-degree (0 to 1)  TODO: Make weighted
        #   5 - out-degree (0 to 1) TODO: Make weighted

        # Initialize node features (shape: num_nodes,6). 
        feature_dict = self.network.get_node_features() 
        
        # Sanity check
        if len(feature_dict) != FeatureDimensions.NODE_FEATURE_COUNT.value: 
            raise ValueError(f"""Mismatch between expected number of 
                             node features ({FeatureDimensions.NODE_FEATURE_COUNT.value})  
                             and number of features in observation ({len(feature_dict)})""")
        
        # Unwrap features 
        # Get the feature dictionaries
        feature_dicts = list(feature_dict.values())


        # Create the feature array
        node_features = np.array([[feature_dict[node_id] for feature_dict in feature_dicts] for node_id in self.network.nodes])
        return {
            FeatureNames.NODE_FEATURES: node_features
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

# Playground for testing
def playground():
    """Test environment with synthetic networks and simulate spread."""
    # Generate a small Erdos-Renyi network
    n_nodes = 20
    network = NetworkGenerator.erdos_renyi(n_nodes, 0.2)
    
    # Initialize environment
    env = InformationSpreadEnv(network, infection_prob=0.3, recovery_prob=0.1, max_steps=10000)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Randomly select initial infection
    initial_infection = np.random.randint(0, n_nodes)
    network.nodes[initial_infection]['state'] = NodeState.INFECTED
    
    print(f"\nNetwork has {len(network.nodes())} nodes and {len(network.edges())} edges")
    print(f"Initially infected node: {initial_infection}")
    
    # Visualization settings
    # visualize_steps = [0, 5, 10, 25, 50, 99]  # Steps to visualize
    max_steps = 1000000
    
    # Metrics to track
    infected_counts = []
    susceptible_counts = []
    
    # Print initial state
    print(f"\nStep 0: {env._get_infected_count()} infected, {env._get_susceptible_count()} susceptible")
    
    # Visualize initial state if needed
    # if 0 in visualize_steps:
    #     env.render()
    
    # Run simulation with random actions
    for step in range(1, max_steps + 1):
        # Generate random actions for both agents
        action_A = np.random.randint(0, n_nodes)
        action_B = np.random.randint(0, n_nodes)
        
        # Execute step
        obs, rewards, done, info = env.step(action_A, action_B)
        
        # Track metrics
        infected_counts.append(env._get_infected_count())
        susceptible_counts.append(env._get_susceptible_count())
        
        # Print current state
        print(f"Step {step}: {env._get_infected_count()} infected, {env._get_susceptible_count()} susceptible")
        
        # Visualize at specified steps
        # if step in visualize_steps:
        #     print(f"\nVisualizing network at step {step}")
        #     #env.render()
        
        # Break if done
        if done:
            print(f"\nSimulation ended at step {step}")
            if env._get_infected_count() == 0:
                print("Infection eradicated")
            elif env._get_susceptible_count() == 0:
                print("All nodes infected")
            else:
                print("Maximum steps reached")
            break
    
    # Plot infection spread over time
    plt.figure(figsize=(10, 6))
    steps = range(len(infected_counts))
    plt.plot(steps, infected_counts, 'r-', label='Infected')
    plt.plot(steps, susceptible_counts, 'b-', label='Susceptible')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Infection Spread Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    playground()