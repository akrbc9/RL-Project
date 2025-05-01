
from InfectionNetwork import NetworkGenerator, NetworkType
from Features import FeatureRegistry
from gymnasium import spaces
from dataclasses import dataclass

@dataclass
class NetworkEnvironmentConfig(): 
    """
        Simple data class to consistently access and 
        define environment parameters.
    """
    
    desc: str = ""
    network_type: NetworkType = NetworkType.ERDOS_RENYI
    nx_n: int = 200 
    nx_m: int = 3 
    nx_k: int = 5
    nx_p: float = 0.2

    num_nodes: int = 400
    num_features: int = 4
    infection_probability: float = 0.5 
    initial_infection_density: float = 0.5
    recovery_probability: float = 0.5
    max_steps: int = 1000 
    max_action_radius: int = 2

    def get_information_network(self): 
        # We don't want to be passing the same network around 
        # everytime we use the same config. 
        if self.network_type == NetworkType.ERDOS_RENYI:
            network = NetworkGenerator.erdos_renyi(
                self.nx_n, self.nx_p
            )
        elif self.network_type == NetworkType.SCALE_FREE:
            network = NetworkGenerator.scale_free(
                self.nx_n, self.nx_m
            )
        elif self.network_type == NetworkType.SMALL_WORLD:
            network = NetworkGenerator.small_world(
                self.nx_n, self.nx_k, self.nx_p
            )
        else: 
            raise ValueError(f'No network "{self.network_type}"')
        return network
    
    def get_action_space(self): 
        return { 
            "agent_a": spaces.Dict({
                "node_id": spaces.Discrete(self.num_nodes) 
            }), 
            "agent_b": spaces.Dict({
                "node_id": spaces.Discrete(self.num_nodes), 
                "radius": spaces.Discrete(self.max_action_radius + 1) # r = 0,1,2. 
            })
        }
    
    def get_observation_space(self): 
        default_obs_space = spaces.Box(low=-1, high=1, 
                                shape=(self.num_nodes, self.num_features))
        return {
            "agent_a": default_obs_space,
            "agent_b": default_obs_space
        }
    
    # We need to_dict and from_dict for 
    # compatibility with RayRLLib API 
    def to_dict(self):
        """Convert config to dictionary for Ray."""
        # Get a copy of the network since we can't directly serialize it
        network_data = None
        if self.get_information_network():
            network_data = {
                'network_type': self.network_type,  # Store network generation info TODO: Dont hardcode this.
                'nx_n': self.nx_n,
                'nx_m': self.nx_m,
                'nx_k': self.nx_k,
                'nx_p': self.nx_p,
            }
        
        return {
            'desc': self.desc,
            'network_data': network_data,
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'infection_probability': self.infection_probability,
            'recovery_probability': self.recovery_probability,
            'initial_infection_density': self.initial_infection_density,
            'max_steps': self.max_steps,
            'max_action_radius': self.max_action_radius
        }
    
    def get_safe_desc(self) -> str: 
        """
            Returns a description without whitespaces (for file naming).
        """
        return self.desc.replace(" ", "")
    
    @classmethod
    def from_dict(cls, config_dict):
        """Reconstruct config from dictionary."""
        # Recreate network if needed
        network = None
        if config_dict.get('network_data'):
            network_data = config_dict['network_data']
        
        return cls(
            network_type=network_data['network_type'],
            nx_n=network_data['nx_n'],
            nx_m=network_data['nx_m'],
            nx_k=network_data['nx_k'],
            nx_p=network_data['nx_p'],

            desc=config_dict['desc'],
            num_nodes=config_dict['num_nodes'],
            num_features=config_dict['num_features'],
            infection_probability=config_dict['infection_probability'],
            initial_infection_density =config_dict['initial_infection_density'],
            recovery_probability=config_dict['recovery_probability'],
            max_steps=config_dict['max_steps'],
            max_action_radius=config_dict['max_action_radius']
        )

# Default number of nodes for all configs
N_NODES = 100

# Configuration 1: Balanced Baseline
# This configuration has balanced parameters between Agent A and Agent B
balanced_config = NetworkEnvironmentConfig(
    network_type=NetworkType.SCALE_FREE,
    nx_n=N_NODES,
    nx_m=3,
    num_nodes=N_NODES,
    num_features=FeatureRegistry.get_feature_count(),
    initial_infection_density=0.2,
    infection_probability=0.6,
    recovery_probability=0.3,
    max_steps=150,
    max_action_radius=1
)

# Configuration 2: High Density Small World
# This configuration models close-knit communities with high connectivity
small_world_config = NetworkEnvironmentConfig(
    network_type=NetworkType.SMALL_WORLD,
    nx_n=N_NODES,
    nx_k=8,  # Higher connectivity
    nx_p=0.1,  # Low rewiring (more clustered)
    num_nodes=N_NODES,
    num_features=FeatureRegistry.get_feature_count(),
    initial_infection_density=0.1,
    infection_probability=0.5,
    recovery_probability=0.2,
    max_steps=150,
    max_action_radius=1
)

# Configuration 3: Random Network
# This configuration uses an Erdos-Renyi random graph
random_network_config = NetworkEnvironmentConfig(
    network_type=NetworkType.ERDOS_RENYI,
    nx_n=N_NODES,
    nx_p=0.05,  # Connection probability
    num_nodes=N_NODES,
    num_features=FeatureRegistry.get_feature_count(),
    initial_infection_density=0.15,
    infection_probability=0.55,
    recovery_probability=0.25,
    max_steps=150,
    max_action_radius=1
)

# Configuration 4: Highly Viral but Easily Recoverable
# Models scenarios where information spreads quickly but doesn't persist
viral_config = NetworkEnvironmentConfig(
    network_type=NetworkType.SCALE_FREE,
    nx_n=N_NODES,
    nx_m=4,  # Higher connectivity
    num_nodes=N_NODES,
    num_features=FeatureRegistry.get_feature_count(),
    initial_infection_density=0.1,
    infection_probability=0.8,  # Very high infection rate
    recovery_probability=0.4,   # High recovery rate
    max_steps=120,
    max_action_radius=1
)

# Configuration 5: Persistent Infection
# Models scenarios where information spreads slowly but is hard to eliminate
persistent_config = NetworkEnvironmentConfig(
    network_type=NetworkType.SCALE_FREE,
    nx_n=N_NODES,
    nx_m=2,  # Lower connectivity
    num_nodes=N_NODES,
    num_features=FeatureRegistry.get_feature_count(),
    initial_infection_density=0.3,
    infection_probability=0.4,  # Lower infection rate
    recovery_probability=0.1,   # Very low recovery rate
    max_steps=200,              # Longer simulations
    max_action_radius=1
)

# Dictionary to easily access configurations by name
EXPERIMENT_CONFIGS = {
    "balanced": balanced_config,
    "small_world": small_world_config,
    "random": random_network_config,
    "viral": viral_config,
    "persistent": persistent_config
}