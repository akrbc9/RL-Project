import ray
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

# Import our custom environment and configurations
from InfectionSpreadEnv import env_creator, InfoSpreadCallbacks
from NetworkEnvironmentConfig import EXPERIMENT_CONFIGS

# Set up directories for storing results
RESULTS_DIR = os.path.join(os.getcwd(), "experiment_results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Training constants
NUM_ITERATIONS = 100
TRAIN_BATCH_SIZE = 4000

def train_policy(config_name, env_config, num_iterations=NUM_ITERATIONS):
    """Train a policy using the specified configuration and save the results."""
    print(f"Starting training for configuration: {config_name}")
    
    # Get the number of features for the observation space
    num_features = env_config.num_features
    
    # Set up the PPO configuration
    config = (
        PPOConfig()
        .environment("network_spread", env_config=env_config.to_dict())
        .multi_agent(
            policies={
                "agent_a": PolicySpec(
                    observation_space=env_config.get_observation_space().get("agent_a"),
                    action_space=env_config.get_action_space().get("agent_a") 
                ),
                "agent_b": PolicySpec(
                    observation_space=env_config.get_observation_space().get("agent_b"),
                    action_space=env_config.get_action_space().get("agent_b") 
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
        )
        .framework("torch")
        .training(
            train_batch_size=TRAIN_BATCH_SIZE,
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
        .reporting(
            keep_per_episode_custom_metrics=True,
        )
    )
    
    # Initialize algorithm
    algo = config.build()
    
    # Storage for metrics
    metrics_history = {
        "infection_rate": [],
        "healing_success_rate": [],
        "largest_component_ratio": [],
        "avg_infection_duration": [],
        "infected_clustering": [],
        "new_infections": [],
        "new_recoveries": [],
        "reward_agent_a": [],
        "reward_agent_b": []
    }
    
    # Training loop with metric collection
    for i in range(num_iterations):
        result = algo.train()
        
        # Extract and store metrics
        print(f"Configuration {config_name}, Iteration {i}:")
        print(f"  Episode reward mean: {result['episode_reward_mean']}")
        
        # Store custom metrics
        for key in metrics_history:
            if key.startswith("reward"):
                agent = key.replace("reward_", "")
                if agent in result['policy_reward_mean']:
                    metrics_history[key].append(
                        result['policy_reward_mean'][agent]
                    )
                else:
                    metrics_history[key].append(0.0)
            elif key in result['custom_metrics']:
                metrics_history[key].append(
                    result['custom_metrics'][key]
                )
            elif key in result.get('hist_stats', {}):
                # Some metrics might be in hist_stats instead
                metrics_history[key].append(np.mean(result['hist_stats'][key]))
            else:
                # If metric isn't found, add None to maintain consistency
                metrics_history[key].append(None)
            print(f"  {key}: {metrics_history[key][-1] if metrics_history[key] else 'N/A'}")
    
    # Save the trained policies
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f"{config_name}")
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Save the metrics history
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f"{config_name}_metrics.csv"), index=False)
    
    # Return the trained algorithm and metrics for further analysis
    return algo, metrics_history

def plot_training_metrics(config_names, metrics_histories):
    """Create visualization plots for the training metrics."""
    # Set up the Seaborn style
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    
    # Create a figure for the plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Infection Rate Comparison
    ax1 = axes[0, 0]
    for config_name in config_names:
        metrics = metrics_histories[config_name]
        ax1.plot(metrics["infection_rate"], label=config_name)
    ax1.set_title("Infection Rate During Training", fontsize=14)
    ax1.set_xlabel("Training Iteration", fontsize=12)
    ax1.set_ylabel("Infection Rate", fontsize=12)
    ax1.legend()
    
    # Plot 2: Agent Rewards Comparison
    ax2 = axes[0, 1]
    for config_name in config_names:
        metrics = metrics_histories[config_name]
        ax2.plot(metrics["reward_agent_a"], linestyle='-', 
                 label=f"{config_name} - Agent A")
        ax2.plot(metrics["reward_agent_b"], linestyle='--', 
                 label=f"{config_name} - Agent B")
    ax2.set_title("Agent Rewards During Training", fontsize=14)
    ax2.set_xlabel("Training Iteration", fontsize=12)
    ax2.set_ylabel("Reward", fontsize=12)
    ax2.legend()
    
    # Plot 3: Healing Success Rate
    ax3 = axes[1, 0]
    for config_name in config_names:
        metrics = metrics_histories[config_name]
        ax3.plot(metrics["healing_success_rate"], label=config_name)
    ax3.set_title("Healing Success Rate", fontsize=14)
    ax3.set_xlabel("Training Iteration", fontsize=12)
    ax3.set_ylabel("Success Rate", fontsize=12)
    ax3.legend()
    
    # Plot 4: Network Structure (Infection Clustering)
    ax4 = axes[1, 1]
    for config_name in config_names:
        metrics = metrics_histories[config_name]
        ax4.plot(metrics["infected_clustering"], label=config_name)
    ax4.set_title("Infection Clustering", fontsize=14)
    ax4.set_xlabel("Training Iteration", fontsize=12)
    ax4.set_ylabel("Clustering Coefficient", fontsize=12)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_comparison.png"))
    plt.close()
    
    # Create individual detailed plots for each configuration
    for config_name in config_names:
        metrics = metrics_histories[config_name]
        
        # Create a figure for this config
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot infection rate 
        ax1 = axes[0, 0]
        ax1.plot(metrics["infection_rate"], label="Infection Rate")
        ax1.set_title(f"{config_name}: Infection Rate", fontsize=14)
        ax1.set_xlabel("Training Iteration", fontsize=12)
        ax1.set_ylabel("Rate", fontsize=12)
        
        # Plot agent rewards
        ax2 = axes[0, 1]
        ax2.plot(metrics["reward_agent_a"], label="Agent A (Spreader)")
        ax2.plot(metrics["reward_agent_b"], label="Agent B (Healer)")
        ax2.set_title(f"{config_name}: Agent Rewards", fontsize=14)
        ax2.set_xlabel("Training Iteration", fontsize=12)
        ax2.set_ylabel("Reward", fontsize=12)
        ax2.legend()
        
        # Plot new infections vs new recoveries
        ax3 = axes[1, 0]
        ax3.plot(metrics["new_infections"], label="New Infections")
        ax3.plot(metrics["new_recoveries"], label="New Recoveries")
        ax3.set_title(f"{config_name}: Infections vs Recoveries", fontsize=14)
        ax3.set_xlabel("Training Iteration", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.legend()
        
        # Plot network structure metrics
        ax4 = axes[1, 1]
        ax4.plot(metrics["largest_component_ratio"], label="Largest Component")
        ax4.plot(metrics["infected_clustering"], label="Infected Clustering")
        ax4.plot(metrics["avg_infection_duration"], label="Avg Infection Duration")
        ax4.set_title(f"{config_name}: Network Structure", fontsize=14)
        ax4.set_xlabel("Training Iteration", fontsize=12)
        ax4.set_ylabel("Value", fontsize=12)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{config_name}_detailed.png"))
        plt.close()

def main():
    """Main function to run the training and analysis."""
    # Initialize Ray
    ray.shutdown()
    ray.init()
    
    # Register the environment with Ray
    register_env("network_spread", env_creator)
    
    # Select configurations to train on
    configs_to_train = ["balanced", "small_world", "viral"]
    
    # Train policies for each selected configuration
    trained_algos = {}
    metrics_histories = {}
    
    for config_name in configs_to_train:
        env_config = EXPERIMENT_CONFIGS[config_name]
        algo, metrics = train_policy(config_name, env_config)
        trained_algos[config_name] = algo
        metrics_histories[config_name] = metrics
    
    # Create comparison plots
    plot_training_metrics(configs_to_train, metrics_histories)
    
    # Shutdown Ray
    ray.shutdown()
    
    print("Training complete! Results are saved in:", RESULTS_DIR)
    print("Checkpoints are saved in:", CHECKPOINTS_DIR)
    print("Plots are saved in:", PLOTS_DIR)

if __name__ == "__main__":
    main()