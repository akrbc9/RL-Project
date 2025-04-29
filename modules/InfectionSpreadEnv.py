import ray
from copy import copy
from ray import tune 
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from InfectionNetwork import NetworkGenerator, NetworkTypes, InformationNetwork, NodeState
from NetworkEnvironmentConfig import * 
from InfectionNetwork import NetworkGenerator, NetworkTypes, InformationNetwork
from Features import * 

from NetworkEnvironmentConfig import NetworkEnvironmentConfig
from gymnasium import spaces
from dataclasses import dataclass
import numpy as np
import random
import networkx as nx



class MultiAgentNetworkEnv(MultiAgentEnv):
    """
        We define a multi-agent wrapper for the environment. 
        This allows us to e.g. decouple observations between agents down the lines. 
        
        This also follows Ray RLLib's best practices for MARL. 
    """
    def __init__(self, net_env_config: NetworkEnvironmentConfig):
        super(MultiAgentNetworkEnv, self).__init__()  # Correct way

    # Initialize with network config
    # Handle either config object or dictionary
        if isinstance(net_env_config, dict):
            self.config = NetworkEnvironmentConfig.from_dict(net_env_config)
        else:
            # Direct object was passed
            self.config = net_env_config

        self.possible_agents = ['agent_a','agent_b'] # Not sure which one I need. 
        self._agent_ids = ['agent_a','agent_b']
        
        # Set up action and observation spaces for agents
        self.action_space = self.config.get_action_space()
        self.observation_space = self.config.get_observation_space()

        # Underlying (Social) Network 
        self.network = self.config.get_information_network()

        # Model Parameters  
        self.initial_infection_density = self.config.initial_infection_density
        self.infection_prob = self.config.infection_probability
        self.recovery_prob = self.config.recovery_probability



        # Simulation Parameters 
        self.max_steps = self.config.max_steps
        self.num_nodes = self.config.num_nodes
        self.current_step = 1 # Just to avoid div by zero
        self.max_action_radius = self.config.max_action_radius
        
        # Initial Infections
        nodes = list(self.network.nodes())
        np.random.shuffle(nodes)
        target_count = int(self.initial_infection_density * self.num_nodes)
        
        for i in range(target_count):
            if i < len(nodes):
                self.network.nodes[nodes[i]]['state'] = NodeState.INFECTED

        # Simulation Data 
        self.current_infection_rate = 0
        self.previous_infection_rate = 0 
        self.node_ids = []
        
    def reset(self, *, seed=None, options=None):
        # Reset all nodes to susceptible
        for node_id in self.network.nodes():
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE
        
        self.current_step = 1
        obs = self._get_observation()

        # return observation dict and infos dict.
        return {
            "agent_a": obs,
            "agent_b": obs
        }, {} # Empty infos dict. 
    
    def step(self, action_dict):
        # Record initial state
        self.previous_infection_rate = self.current_infection_rate 

        # Unpack actions from both agents
        action_a = action_dict["agent_a"]
        action_b = action_dict["agent_b"]
        
        # First execute Agent A's action (infection) 
        # & Agent B's action (prevention)
        self._execute_action_a(action_a)
        self._execute_action_b(action_b)

        # Then process information propagation 
        # Returns array of all nodes that were infected 
        self.newly_infected_nodes = self._propagate_information() 

        # Then process recovery 
        self.newly_recovered_nodes = self._process_recovery()

        # Update step counter += 1 
        self.current_step += 1 

        # Update infection lengths
        for node in self.network.nodes:
            if self.network.nodes[node]['state'] == NodeState.INFECTED: 
                self.network.nodes[node]['length_of_infection'] +=1 

        # Update edge weights 
        # 5% edge weight increase
        for u, v in self.network.edges():
            if (self.network.nodes[u]['state'] == NodeState.INFECTED and 
                self.network.nodes[v]['state'] == NodeState.INFECTED):
                # Reinforcement of connections between infected nodes
                self.network[u][v]['weight'] = min(1.0, self.network[u][v]['weight'] * 1.05)
        # Update data 
        self.last_infection_rate = self.current_infection_rate
        self.current_infection_rate = self._get_infection_rate()

        # Get observation 
        # This will be the feature matrix of this network 
        obs = self._get_observation()

        # Calculate Rewards 
        reward_a = self._calculate_reward_a()
        reward_b = self._calculate_reward_b()
        
        # Check if the episode is done 
        done = self._is_done() 

        # Get custom metrics
        custom_metrics = self._get_episode_info()
        
        # Add metrics to info dict
        infos = {
            "agent_a": {"metrics": custom_metrics},
            "agent_b": {"metrics": custom_metrics}
        }
            
        # Format for multi-agent environment
        observations = {
            "agent_a": obs,
            "agent_b": obs
        }
        rewards = {
            "agent_a": reward_a,
            "agent_b": reward_b
        }
        terminateds = {
            "agent_a": done,
            "agent_b": done,
            "__all__": done
        }
        
        # New Gymnasium API requires "truncateds" to indicate time limit cutoffs
        # (false if terminated normally, true if cut off by time limit)
        # We don't use this so just set all to false. 
        truncateds = {
            "agent_a": False,
            "agent_b": False,
            "__all__": False
        }
        
        return observations, rewards, terminateds, truncateds, infos

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
    
    # Local model controller functions  
    def _execute_action_a(self, action): 
        """
            Execute Agent A's action (infection).
            Args: 
                action: ID of node to infect.
        """
        node_id = action['node_id']

        if node_id not in self.network: 
            raise ValueError(f"Invalid node {node_id} passed to Agent A (Spreader")
        
        node_data = self.network.nodes[node_id]
    
        # If node is susceptible and infection threshold is reached:
        # Skeptical nodes are less likely to be infected. 
        eff_prob = self.infection_prob * (1 - node_data['skepticism'])
        self._infect_node(node_id, eff_prob)

    def _execute_action_b(self, action): 
        radius = min(action['radius'], self.max_action_radius)
        center_node = action['node_id']

        if center_node not in self.network.nodes: 
            raise ValueError(f"Invalid node {center_node} passed to Agent B (Healer)")
                
        # Identify the subnetwork (BFS to specified radius)
        if radius == 0: 
            self._heal_node(center_node, self.recovery_prob) # TODO: Add Community effects to recovery.

        elif radius > 0:
            subnetwork = set([center_node])
            frontier = set([center_node])
            for r in range(radius):
                next_frontier = set()
                for node in frontier:
                    for succ_node in self.network.successors(node):
                        # Slow down infection spread within a 2 step redius. 
                        # This is also weighted by radius 
                        self.network[node][succ_node]['weight'] *= 0.85 * (3 - r)/3

                        # Do the same for skepticism 
                        new_skept = min(
                          1, 
                          self.network.nodes[node]['skepticism'] * (1 + 0.10*(self.max_action_radius - r)/self.max_action_radius)
                          )
                        self.network.nodes[node]['skepticism'] = new_skept


                    # Add neighbors that aren't already in subnetwork
                    next_frontier.update(
                        [n for n in self.network.successors(node) 
                        if n not in subnetwork]
                    )
                subnetwork.update(next_frontier)
                frontier = next_frontier

        # TODO: Max on size of subnetwork? 

    def _propagate_information(self):
        """
            Propagate information through the network using SIR model.
            Returns: 
                [node_id]: All newly infected nodes 
            
            # TODO: Include community based propagation rates
        """
        new_infections = []

        # Infection process
        # For each infected node
        for node_id in self.network.nodes:
            if self.network.nodes[node_id]['state'] != NodeState.INFECTED:
                continue
            
            skepticism = self.network.nodes[node_id]['skepticism']

            # Attempt to infect each susceptible neighbor
            for neighbor_id in self.network.successors(node_id):
                if self.network.nodes[neighbor_id]['state'] != NodeState.SUSCEPTIBLE:
                    continue
                
                # Use edge weight to influence probability
                edge_weight = self.network[node_id][neighbor_id]['weight']
                eff_prob = self.infection_prob * (1 - skepticism) * edge_weight

                # Attempt infection
                if self._infect_node(neighbor_id, eff_prob):
                    new_infections.append(neighbor_id) 
        
        return new_infections
    
    def _process_recovery(self):
        """
            Propagate recovery through neighboring edges 
            Returns: 
                [node_id]: All newly recovered nodes 
            TODO Implement the network recovery logic
        """
        recoveries = []
        # Recovery process
        # For each infected node, there's a chance to recover
        for node_id in self.network.nodes:
            if (self.network.nodes[node_id]['state'] == NodeState.INFECTED 
                and node_id not in self.newly_infected_nodes):
                duration = self.network.nodes[node_id]['length_of_infection']
                eff_recovery_prob = self.recovery_prob /(1 + 0.1 * duration)
                if self._heal_node(node_id, eff_recovery_prob): 
                    recoveries.append(node_id)
        
        return recoveries

    def _get_observation(self):
        """
            Generate observation based on current network state.

            Returns: 
                ndarray of shape (num_nodes, num_features) 
        """
        # Node IDs is a mapping of nodes to their respective indices
        feature_matrix, node_ids = FeatureRegistry.extract_feature_matrix(self.network)
        self.node_ids = node_ids
        
        return feature_matrix

    def _calculate_reward_a(self):
        infection_change = self.current_infection_rate - self.previous_infection_rate
        
        # Base reward on infection growth
        growth_reward = infection_change * 20.0  # Scaling factor
        
        # Add small reward for maintaining high infection
        persistence_reward = self.current_infection_rate * 2.0
        
        return growth_reward + persistence_reward 
          

    def _calculate_reward_b(self):
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
        
        susceptible_count = self._get_susceptible_count()
        infected_count = self._get_infected_count()
        
        if (self._get_susceptible_count() == 0 
            or self._get_infected_count() == 0):
            return True
        
        return False
    
    def _get_episode_info(self):
        """Calculate episode metrics specific to information spread."""
        # Network state metrics
        infected_count = self._get_infected_count()
        infection_rate = infected_count / self.num_nodes
        
        # Calculate clustering of infected nodes
        infected_nodes = [n for n, d in self.network.nodes(data=True) 
                        if d['state'] == NodeState.INFECTED]
        infected_clustering = 0
        if len(infected_nodes) > 2:
            infected_subgraph = self.network.subgraph(infected_nodes)
            infected_clustering = nx.transitivity(infected_subgraph)
        
        # Calculate average infection duration
        avg_infection_duration = np.mean([
            self.network.nodes[n].get('infection_length', 0) 
            for n in infected_nodes
        ]) if infected_nodes else 0
        
        # Structural metrics
        largest_component_size = 0
        if infected_nodes:
            infected_subgraph = self.network.subgraph(infected_nodes)
            largest_cc = max(nx.connected_components(infected_subgraph.to_undirected()), 
                            key=len)
            largest_component_size = len(largest_cc) / len(infected_nodes)
        
        # Intervention effectiveness
        healing_success_rate = (len(self.newly_recovered_nodes) / 
                            max(1, self._get_infected_count() + len(self.newly_recovered_nodes)))
        
        # Return dictionary of metrics
        return {
            "infection_rate": infection_rate,
            "infected_clustering": infected_clustering,
            "avg_infection_duration": avg_infection_duration,
            "largest_component_ratio": largest_component_size,
            "healing_success_rate": healing_success_rate,
            "new_infections": len(self.newly_infected_nodes),
            "new_recoveries": len(self.newly_recovered_nodes)
        }    

    def _get_infected_count(self):
        """
            Get the number of infected nodes.
        """
        return sum(1 for _, data in self.network.nodes(data=True) 
                  if data['state'] == NodeState.INFECTED)
    
    def _get_susceptible_count(self):
        """
            Get the number of susceptible nodes.
        """
        return sum(1 for _, data in self.network.nodes(data=True) 
                  if data['state'] == NodeState.SUSCEPTIBLE)
    
    def _get_infection_rate(self): 
        return self._get_infected_count()/self.num_nodes
    
    def _heal_node(self, node_id: int, prob: float) -> bool:
        """
            Heal nodes in node_ids with probability prob. 
            Returns: 
                success (True) or failure (False)
        """
        if node_id not in self.network.nodes: 
            raise ValueError(f'Node {node_id} not found.')
        
        if random.random() < prob and self.network.nodes[node_id]['state'] == NodeState.INFECTED:     
            self.network.nodes[node_id]['length_of_infection'] = 0 
            self.network.nodes[node_id]['state'] = NodeState.SUSCEPTIBLE

            # Strongly reduce weights of infected nodes. 
            for pred in self.network.predecessors(node_id): 
                self.network[pred][node_id]['weight'] *= 0.1 
            return True 
        
        return False

    def _infect_node(self, node_id: np.ndarray, prob: float) -> bool: 
        """
            Infect nodes in node_ids with probability prob. 
            Returns: 
                successful infection (True) or failure (False)
        """       
        if node_id not in self.network.nodes: 
            raise ValueError(f'Node {node_id} not found.')
        
        if random.random() < prob and self.network.nodes[node_id]['state'] == NodeState.SUSCEPTIBLE: 
            self.network.nodes[node_id]['state'] = NodeState.INFECTED
            return True 
        
        return False
                
    def render(self):
        """Render the environment."""
        self.network.visualize()

def env_creator(env_config):
    return MultiAgentNetworkEnv(env_config)

class InfoSpreadCallbacks(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        """Called on each episode step."""
        # Safe way to access environment
        infos = episode.last_info_for("agent_a")
        try:
            for key,val in infos["metrics"].items():     
                # Initialize if needed
                if key not in episode.user_data:
                    episode.user_data[key] = []
                
                # Store current rate
                episode.user_data[key].append(val)
                
                # Add as custom metric
                episode.custom_metrics[key] = val
            
        
        except Exception as e:
            print(f"Error in callback: {e}")
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at the end of an episode."""
        try:
            # Get infection rates from episode data
            infection_rates = episode.user_data.get("infection_rates", [])
            
            if infection_rates:
                # Calculate metrics
                episode.custom_metrics["infection_auc"] = sum(infection_rates) / len(infection_rates)
                episode.custom_metrics["final_infection_rate"] = infection_rates[-1]
                
                # Add change from start to end if we have multiple points
                if len(infection_rates) > 1:
                    episode.custom_metrics["infection_change"] = infection_rates[-1] - infection_rates[0]
        except Exception as e:
            print(f"Error in episode end callback: {e}")

        