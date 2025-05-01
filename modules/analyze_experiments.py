import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

# Import our custom environment and configurations
from InfectionSpreadEnv import env_creator, MultiAgentNetworkEnv
from NetworkEnvironmentConfig import EXPERIMENT_CONFIGS
from NetworkEnvironmentConfig import NetworkEnvironmentConfig
from Features import FeatureRegistry, NodeFeatures

# Setup directories
RESULTS_DIR = os.path.join(os.getcwd(), "experiment_results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories if they don't exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Number of evaluation episodes
NUM_EPISODES = 50


def load_trained_policy(config_name):
    """Load a trained policy from a checkpoint."""
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, config_name)
    
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_")]
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    # Create config from the saved configuration
    config = EXPERIMENT_CONFIGS[config_name]
    
    # Load the algorithm
    algo = PPO(
        env="network_spread",
        config={
            "env_config": config.to_dict(),
            "framework": "torch",
            "multiagent": {
                "policies_to_train": ["agent_a", "agent_b"],
                "policies": {
                    "agent_a": (None, config.get_observation_space()["agent_a"], 
                               config.get_action_space()["agent_a"], {}),
                    "agent_b": (None, config.get_observation_space()["agent_b"], 
                               config.get_action_space()["agent_b"], {})
                },
                "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
            }
        }
    )
    
    # Restore from checkpoint
    algo.restore(checkpoint_path)
    
    return algo, config


def run_episode(env, algo, track_actions=False):
    """Run a single episode with the trained policy and return episode data."""
    # Reset environment
    obs, _ = env.reset()
    done = {"__all__": False}
    
    # Initialize episode data
    episode_data = {
        "infection_rate": [],
        "infected_nodes": [],
        "susceptible_nodes": [],
        "new_infections": [],
        "new_recoveries": [],
        "agent_a_reward": 0,
        "agent_b_reward": 0,
        "agent_a_actions": [] if track_actions else None,
        "agent_b_actions": [] if track_actions else None,
        "agent_b_radius_choices": [] if track_actions else None
    }
    
    # Run episode until done
    while not done["__all__"]:
        # Get actions from policies
        actions = {}
        for agent_id in obs:
            action = algo.compute_single_action(obs[agent_id], policy_id=agent_id)
            
            # Convert to format expected by environment
            if agent_id == "agent_a":
                actions[agent_id] = {"node_id": action}
                if track_actions:
                    episode_data["agent_a_actions"].append(action)
            else:  # agent_b
                node_id, radius = action
                actions[agent_id] = {"node_id": node_id, "radius": radius}
                if track_actions:
                    episode_data["agent_b_actions"].append(node_id)
                    episode_data["agent_b_radius_choices"].append(radius)
        
        # Take step in environment
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        done = terminateds
        
        # Update episode data
        episode_data["infection_rate"].append(env.current_infection_rate)
        episode_data["infected_nodes"].append(env._get_infected_count())
        episode_data["susceptible_nodes"].append(env._get_susceptible_count())
        episode_data["new_infections"].append(len(env.newly_infected_nodes))
        episode_data["new_recoveries"].append(len(env.newly_recovered_nodes))
        
        # Accumulate rewards
        episode_data["agent_a_reward"] += rewards["agent_a"]
        episode_data["agent_b_reward"] += rewards["agent_b"]
    
    return episode_data


def evaluate_policy(config_name, num_episodes=NUM_EPISODES):
    """Evaluate a trained policy over multiple episodes."""
    print(f"Evaluating policy for configuration: {config_name}")
    
    # Load trained policy
    algo, config = load_trained_policy(config_name)
    
    # Create environment
    env = MultiAgentNetworkEnv(config)
    
    # Run episodes
    all_episode_data = []
    
    for i in tqdm(range(num_episodes)):
        # Track actions for some episodes
        track_actions = (i % 5 == 0)  # Track every 5th episode
        episode_data = run_episode(env, algo, track_actions)
        all_episode_data.append(episode_data)
    
    # Aggregate data across episodes
    aggregated_data = {
        "infection_rate": [],
        "infected_nodes": [],
        "susceptible_nodes": [],
        "new_infections": [],
        "new_recoveries": [],
        "agent_a_rewards": [],
        "agent_b_rewards": [],
        "agent_b_radius_distribution": [0, 0, 0]  # For radius 0, 1, 2
    }
    
    max_steps = max([len(ep["infection_rate"]) for ep in all_episode_data])
    
    # Initialize arrays
    for key in ["infection_rate", "infected_nodes", "susceptible_nodes", 
                "new_infections", "new_recoveries"]:
        aggregated_data[key] = np.zeros((num_episodes, max_steps))
        aggregated_data[key].fill(np.nan)
    
    # Fill arrays with episode data
    for i, ep_data in enumerate(all_episode_data):
        aggregated_data["agent_a_rewards"].append(ep_data["agent_a_reward"])
        aggregated_data["agent_b_rewards"].append(ep_data["agent_b_reward"])
        
        for j, val in enumerate(ep_data["infection_rate"]):
            aggregated_data["infection_rate"][i, j] = val
            aggregated_data["infected_nodes"][i, j] = ep_data["infected_nodes"][j]
            aggregated_data["susceptible_nodes"][i, j] = ep_data["susceptible_nodes"][j]
            aggregated_data["new_infections"][i, j] = ep_data["new_infections"][j]
            aggregated_data["new_recoveries"][i, j] = ep_data["new_recoveries"][j]
        
        # Count radius choices
        if ep_data["agent_b_radius_choices"] is not None:
            for radius in ep_data["agent_b_radius_choices"]:
                if radius < len(aggregated_data["agent_b_radius_distribution"]):
                    aggregated_data["agent_b_radius_distribution"][radius] += 1
    
    # Save the aggregated data
    np.save(os.path.join(ANALYSIS_DIR, f"{config_name}_evaluation.npy"), aggregated_data)
    
    return aggregated_data


def analyze_agent_b_actions(config_name, aggregated_data):
    """Analyze the actions taken by Agent B (Healer)."""
    # Analyze radius choices
    radius_distribution = aggregated_data["agent_b_radius_distribution"]
    total_choices = sum(radius_distribution)
    
    if total_choices > 0:
        radius_percentages = [count / total_choices * 100 for count in radius_distribution]
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Radius 0', 'Radius 1', 'Radius 2'], y=radius_percentages)
        plt.title(f"Agent B Radius Choices Distribution - {config_name}", fontsize=14)
        plt.xlabel("Action Radius", fontsize=12)
        plt.ylabel("Percentage of Actions (%)", fontsize=12)
        
        # Add percentage labels
        for i, p in enumerate(radius_percentages):
            plt.text(i, p + 1, f"{p:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{config_name}_radius_distribution.png"))
        plt.close()
    
    return radius_distribution


def plot_infection_trajectories(config_names, all_data):
    """Create plots comparing infection trajectories across configurations."""
    # Set Seaborn style
    sns.set(style="whitegrid", context="paper", palette="colorblind")
    
    # Create a figure for infection rates
    plt.figure(figsize=(12, 8))
    
    for config_name in config_names:
        data = all_data[config_name]
        infection_rates = data["infection_rate"]
        
        # Calculate mean and percentiles
        mean_rates = np.nanmean(infection_rates, axis=0)
        p25 = np.nanpercentile(infection_rates, 25, axis=0)
        p75 = np.nanpercentile(infection_rates, 75, axis=0)
        
        # Plot mean with fill between for 25-75 percentile range
        x = np.arange(len(mean_rates))
        plt.plot(x, mean_rates, label=config_name)
        plt.fill_between(x, p25, p75, alpha=0.3)
    
    plt.title("Infection Rate Trajectories Across Configurations", fontsize=14)
    plt.xlabel("Simulation Step", fontsize=12)
    plt.ylabel("Infection Rate", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "infection_trajectories_comparison.png"))
    plt.close()
    
    # Create a heatmap of infection rates over time
    for config_name in config_names:
        data = all_data[config_name]
        infection_rates = data["infection_rate"]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(infection_rates, cmap="YlOrRd", vmin=0, vmax=1, 
                   cbar_kws={'label': 'Infection Rate'})
        plt.title(f"Infection Rate Heatmap - {config_name}", fontsize=14)
        plt.xlabel("Simulation Step", fontsize=12)
        plt.ylabel("Episode", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{config_name}_infection_heatmap.png"))
        plt.close()


def plot_new_infections_vs_recoveries(config_names, all_data):
    """Plot new infections vs recoveries for each configuration."""
    for config_name in config_names:
        data = all_data[config_name]
        
        # Calculate mean infections and recoveries over episodes
        mean_infections = np.nanmean(data["new_infections"], axis=0)
        mean_recoveries = np.nanmean(data["new_recoveries"], axis=0)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(mean_infections))
        plt.plot(x, mean_infections, label="New Infections", color="red")
        plt.plot(x, mean_recoveries, label="New Recoveries", color="blue")
        plt.fill_between(x, np.zeros_like(mean_infections), mean_infections, color="red", alpha=0.2)
        plt.fill_between(x, np.zeros_like(mean_recoveries), mean_recoveries, color="blue", alpha=0.2)
        
        plt.title(f"New Infections vs Recoveries - {config_name}", fontsize=14)
        plt.xlabel("Simulation Step", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{config_name}_infections_vs_recoveries.png"))
        plt.close()


def analyze_network_structure(config_names, all_data):
    """Analyze the network structure metrics."""
    # Comparison plot for all configs
    plt.figure(figsize=(12, 8))
    
    for config_name in config_names:
        data = all_data[config_name]
        mean_rates = np.nanmean(data["infection_rate"], axis=0)
        x = np.arange(len(mean_rates))
        plt.plot(x, mean_rates, label=f"{config_name}")
    
    plt.title("Infection Rate by Network Type", fontsize=14)
    plt.xlabel("Simulation Step", fontsize=12)
    plt.ylabel("Infection Rate", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "network_type_comparison.png"))
    plt.close()


def compare_agent_performance(config_names, all_data):
    """Compare agent performance across configurations."""
    # Create dataframe for easier plotting
    performance_data = []
    
    for config_name in config_names:
        data = all_data[config_name]
        
        for i in range(len(data["agent_a_rewards"])):
            performance_data.append({
                "Configuration": config_name,
                "Agent": "Agent A (Spreader)",
                "Reward": data["agent_a_rewards"][i]
            })
            performance_data.append({
                "Configuration": config_name,
                "Agent": "Agent B (Healer)",
                "Reward": data["agent_b_rewards"][i]
            })
    
    df = pd.DataFrame(performance_data)
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Configuration", y="Reward", hue="Agent", data=df)
    plt.title("Agent Performance Across Configurations", fontsize=14)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Total Episode Reward", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "agent_performance_comparison.png"))
    plt.close()


def main():
    """Run the analysis pipeline."""
    # Register environment
    register_env("network_spread", env_creator)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Configurations to analyze
    config_names = ["balanced", "small_world", "viral"]
    
    # Evaluate policies and collect data
    all_data = {}
    for config_name in config_names:
        all_data[config_name] = evaluate_policy(config_name)
    
    # Analyze actions
    for config_name in config_names:
        analyze_agent_b_actions(config_name, all_data[config_name])
    
    # Generate plots
    plot_infection_trajectories(config_names, all_data)
    plot_new_infections_vs_recoveries(config_names, all_data)
    analyze_network_structure(config_names, all_data)
    compare_agent_performance(config_names, all_data)
    
    # Shutdown Ray
    ray.shutdown()
    
    print("Analysis complete! All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()