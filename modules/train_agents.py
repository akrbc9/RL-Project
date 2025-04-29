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

from gymnasium import spaces
from dataclasses import dataclass

# Custom Modules
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx 

from NetworkEnvironmentConfig import * 
from InfectionNetwork import NetworkGenerator, NetworkTypes, InformationNetwork
from Features import * 
from InfectionSpreadEnv import *

# 0. Define Constants  
N_NODES = 50 
INFECTION_PROB = 0.5 
RECOVERY_PROB = 0.05
MAX_STEPS = 100
NUM_ITERATIONS = 100

# 1. Initialize Ray
ray.shutdown()
ray.init()

# Register the environment with Ray
register_env("network_spread", env_creator)

# Create training environment config 
train_env_config = NetworkEnvironmentConfig(
    network_type = "scale_free", 
    nx_n = N_NODES,
    nx_m = 3,
    num_nodes = N_NODES, 
    num_features = FeatureRegistry.get_feature_count(),
    initial_infection_density = 0.5,
    infection_probability = INFECTION_PROB, 
    recovery_probability = RECOVERY_PROB, 
    max_steps = MAX_STEPS, 
    max_action_radius = 1 
)

# Prepare to pass to configuration for RLLib
env_config_dict = train_env_config.to_dict()

# Then use its observation spaces in the config
config = (
    PPOConfig()
    .environment("network_spread", env_config=env_config_dict)
    .multi_agent(
        policies={
            "agent_a": PolicySpec(
                observation_space=train_env_config.get_observation_space().get("agent_a"),
                action_space=train_env_config.get_action_space().get("agent_a") 
            ),
            "agent_b": PolicySpec(
                observation_space=train_env_config.get_observation_space().get("agent_b"),
                action_space=train_env_config.get_action_space().get("agent_b") 
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
        entropy_coeff=0.2,
    )
    .rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=100
    )
    .debugging(log_level="WARNING")
    .callbacks(InfoSpreadCallbacks)

    # Add TensorBoard for visualization
    .reporting(
        keep_per_episode_custom_metrics=True,
    )
)

# Run training with monitoring
def stats_along_columns(inhomogenous_array):
    """ 
        Utility functions to get averages for each timstep across episodes. 
        Since some episodes end earlier than other, we can't use standard numpy functions
        because the arrays are inhomogenous.
    """
    max_length = max([len(row) for row in inhomogenous_array])
    means = np.zeros(max_length)
    medians = np.zeros(max_length)
    stds = np.zeros(max_length) 

    for col in range(max_length):
        col_vals = []
        for row in inhomogenous_array:
            if len(row) >= col + 1: 
                col_vals.append(row[col])
        if col_vals:
            means[col] = np.mean(col_vals)
            medians[col] = np.median(col_vals)
            stds[col] = np.std(col_vals)

    return {
        'means': means,
        'medians': medians, 
        'stds': stds 
    }

# Initialize algorithm
algo = config.build()

# Storage for metrics
metrics_history = {
    "infection_rate": [],
    "healing_success_rate": [],
    "largest_component_ratio": [],
    "avg_infection_duration": [],
    "reward_agent_a": [],
    "reward_agent_b": []
}

# Training loop with metric collection
for i in range(NUM_ITERATIONS):
    result = algo.train()
    
    # Extract and store metrics
    print(f"Iteration {i}:")
    print(f"  Episode reward mean: {result['episode_reward_mean']}")
    
    # Store custom metrics
    for key in metrics_history:
        if key.startswith("reward"):
            agent = key.replace("reward_", "")
            print(key)
            metrics_history[key].append(
                result['policy_reward_mean'][agent]
            )
        elif key in result['custom_metrics']:
            metrics_history[key].append(
                result['custom_metrics'][key]
            )
        print(f"  {key}: {metrics_history[key][-1] if metrics_history[key] else 'N/A'}")
    
# Create visualizations
plt.figure(figsize=(15, 10))
infection_rate_trajectory_stats = stats_along_columns(metrics_history["infection_rate"])
healing_success_rate_trajectory_stats = stats_along_columns(metrics_history["healing_success_rate"])

# Plot 1: Infection rate over time
plt.subplot(2, 2, 1)
plt.plot(infection_rate_trajectory_stats['means'])
plt.title("Infection Rate During Training")
plt.xlabel("Training Iteration")
plt.ylabel("Infection Rate")
plt.grid(True)

# Plot 2: Healing success rate
plt.subplot(2, 2, 2)
plt.plot(healing_success_rate_trajectory_stats['means'])
plt.title("Healing Success Rate")
plt.xlabel("Training Iteration")
plt.ylabel("Success Rate")
plt.grid(True)

# Plot 3: Rewards for both agents
plt.subplot(2, 2, 3)
plt.plot(metrics_history["reward_agent_a"], label="Agent A (Spreader)")
plt.plot(metrics_history["reward_agent_b"], label="Agent B (Healer)")
plt.title("Agent Rewards During Training")
plt.xlabel("Training Iteration")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
    
    # Plot 4: Network structure metrics
    # plt.subplot(2, 2, 4)
    # plt.plot(np.mean(metrics_history["largest_component_ratio"], axis=1), label="Largest Component")
    # plt.plot(np.mean(metrics_history["avg_infection_duration"], axis=1), label="Avg Infection Duration")
    # plt.title("Network Structure Metrics")
    # plt.xlabel("Training Iteration")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)
    
plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()

# Save the trained policies
checkpoint_path = algo.save()
print(f"Saved checkpoint to: {checkpoint_path}")
