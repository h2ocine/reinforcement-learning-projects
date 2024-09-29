import optuna
import numpy as np
import matplotlib.pyplot as plt
from functools import partial  # Permet de passer des arguments à une fonction objective
import numpy as np
import os
from typing import List

import hydra
import optuna
import yaml
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
import gymnasium as gym

from bbrl.utils.chrono import Chrono

import matplotlib
import matplotlib.pyplot as plt

from mazemdp.toolbox import sample_categorical
from mazemdp.mdp import Mdp
from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from functools import partial
from typing import Dict, Tuple, Any


matplotlib.use("TkAgg")

# For visualization
os.environ["VIDEO_FPS"] = "5"
if not os.path.isdir("./videos"):
    os.mkdir("./videos")

from IPython.display import Video


def naive_actor_critic(
    env: MazeMDPEnv,
    alpha_actor: float,
    alpha_critic: float,
    gamma: float,
    nb_episodes: int = 1000,
    timeout: int = 50,
    render: bool = False,
):
    """
    Naive actor-critic algorithm
    Args:
        env: The environment
        alpha_actor: Learning rate for the actor
        alpha_critic: Learning rate for the critic
        gamma: Discount factor
        nb_episodes: Number of episodes
        timeout: Maximum number of steps per episode
        render: Boolean flag to render the environment
    Returns:
        policy, value_function, trajectories
    """
    # Initialize value function V(s) and the policy pi(a|s)
    V = np.zeros(env.nb_states)  # Critic (value function V(s))
    pi = np.ones((env.nb_states, env.action_space.n)) / env.action_space.n  # Actor (policy pi(a|s))

    # Function to renormalize the policy for a given state
    def renormalize_policy(pi, s):
        pi[s, :] = pi[s, :] / np.sum(pi[s, :])

    trajectories = []

    for episode in range(nb_episodes):
        s, _ = env.reset(uniform=True)
        cpt = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and cpt < timeout:
            if render:
                env.draw_v_pi(V, pi.argmax(axis=1))

            # Sample an action from the current policy pi(a|s)
            a = sample_categorical(pi[s, :])

            # Perform a step in the environment
            s_next, r, terminated, truncated, _ = env.step(a)

            # Calculate the temporal difference error (TD error)
            delta = r + gamma * V[s_next] * (1 - terminated) - V[s]

            # Update the critic (value function)
            V[s] = V[s] + alpha_critic * delta

            # Update the actor (policy) for the action taken
            pi_temp = pi[s, a] + alpha_actor * delta
            pi_temp = max(pi_temp, 1e-8)  # Ensure non-negative probability

            # Apply renormalization to ensure sum of probabilities is 1
            pi[s, a] = pi_temp
            renormalize_policy(pi, s)

            # Move to the next state
            s = s_next
            cpt += 1

        trajectories.append(cpt)

    return pi, V, trajectories

def naive_actor_critic_dif(
    env: MazeMDPEnv,
    alpha_actor: float,
    alpha_critic: float,
    gamma: float,
    nb_episodes: int = 1000,
    timeout: int = 50,
    render: bool = False,
):
    """
    Naive actor-critic algorithm.
    Returns:
        policy, value_function, trajectories, cumulative_rewards, entropies
    """
    # Initialize value function V(s) and the policy pi(a|s)
    V = np.zeros(env.nb_states)  # Critic (value function V(s))
    pi = np.ones((env.nb_states, env.action_space.n)) / env.action_space.n  # Actor (policy pi(a|s))

    # Function to renormalize the policy for a given state
    def renormalize_policy(pi, s):
        pi[s, :] = pi[s, :] / np.sum(pi[s, :])

    trajectories = []
    cumulative_rewards = []
    entropies = []  # To track the entropy of the policy over episodes

    for episode in range(nb_episodes):
        s, _ = env.reset(uniform=True)
        cpt = 0
        terminated = False
        truncated = False
        episode_reward = 0  # Initialize reward for this episode
        episode_entropy = 0  # Initialize entropy for this episode

        while not (terminated or truncated) and cpt < timeout:
            if render:
                env.draw_v_pi(V, pi.argmax(axis=1))

            # Sample an action from the current policy pi(a|s)
            a = sample_categorical(pi[s, :])

            # Perform a step in the environment
            s_next, r, terminated, truncated, _ = env.step(a)

            # Accumulate reward
            episode_reward += r

            # Calculate the temporal difference error (TD error)
            delta = r + gamma * V[s_next] * (1 - terminated) - V[s]

            # Update the critic (value function)
            V[s] = V[s] + alpha_critic * delta

            # Update the actor (policy) for the action taken
            pi_temp = pi[s, a] + alpha_actor * delta
            pi_temp = max(pi_temp, 1e-8)  # Ensure non-negative probability

            # Apply renormalization to ensure sum of probabilities is 1
            pi[s, a] = pi_temp
            renormalize_policy(pi, s)

            # Calculate entropy for the current policy (at state s)
            entropy = -np.sum(pi[s, :] * np.log(pi[s, :] + 1e-8))
            episode_entropy += entropy  # Accumulate entropy for the episode

            # Move to the next state
            s = s_next
            cpt += 1

        trajectories.append(cpt)
        cumulative_rewards.append(episode_reward)  # Store cumulative reward for the episode
        entropies.append(episode_entropy / cpt)  # Store average entropy for the episode

    return pi, V, trajectories, cumulative_rewards, entropies

def create_maze_from_params(ac_params):
    """
    Creates a maze environment using parameters from ac_params.
    
    Args:
        ac_params: Dictionary containing all relevant parameters for the maze environment.
    
    Returns:
        env: The initialized maze environment.
    """
    env_params = ac_params['mdp']

    env = gym.make(
    "MazeMDP-v0",
    kwargs={"width": env_params['width'], "height": env_params['height'], "ratio": env_params['ratio'], "hit": 0.0},
    render_mode=env_params['render_mode'],
)
    env.reset()
    #env.unwrapped.init_draw("The maze")
    return env

def objective(trial: optuna.Trial, 
              ac_params: Dict[str, Any], 
              learning_curves_dict: Dict[Tuple[float, float], np.ndarray], 
              n_runs: int = 5) -> float:
    """
    Objective function for Optuna to optimize. Runs the actor-critic algorithm multiple times
    and returns the mean norm of the value function across those runs.

    Args:
        trial: Optuna trial object to suggest hyperparameters.
        ac_params: Dictionary containing actor-critic parameters.
        learning_curves_dict: Dictionary to store learning curves for each trial.
        n_runs: Number of runs to average for each hyperparameter set.
    
    Returns:
        mean_value_norm: The mean norm of the value function across multiple runs.
    """
    # Sample alpha_actor and alpha_critic 
    alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
    alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
    
    # Get the parameters for the actor-critic algorithm
    nb_episodes = ac_params['nb_episodes']
    timeout = ac_params['timeout']
    gamma = ac_params['gamma']
    
    # Run multiple experiments
    total_value_norm = 0
    combined_trajectories = np.zeros(nb_episodes)
    for _ in range(n_runs):
        env = create_maze_from_params(ac_params)  # Assure-toi que cette fonction est bien définie ou importée
        _, V, trajectories = naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
        total_value_norm += np.linalg.norm(V)
        combined_trajectories += np.array(trajectories)
    
    learning_curves_dict[(alpha_actor, alpha_critic)] = combined_trajectories / n_runs
    mean_value_norm = total_value_norm / n_runs
    #Return the negative mean value norm (as we minimize in Optuna -> maximize the mean value norms)
    return -1 * mean_value_norm

def objective_cumulative_reward(trial: optuna.Trial, 
                                ac_params: Dict[str, Any], 
                                learning_curves_dict: Dict[Tuple[float, float], np.ndarray], 
                                n_runs: int = 5) -> float:
    """
    Objective function that optimizes the cumulative reward across multiple runs.
    """
    # Sample alpha_actor and alpha_critic 
    alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
    alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
    
    # Get actor-critic parameters
    nb_episodes = ac_params['nb_episodes']
    timeout = ac_params['timeout']
    gamma = ac_params['gamma']
    
    total_cumulative_reward = 0
    combined_trajectories = np.zeros(nb_episodes)
    
    for _ in range(n_runs):
        env = create_maze_from_params(ac_params)
        _, _, trajectories, cumulative_rewards,_ = naive_actor_critic_dif(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
        
        total_cumulative_reward += sum(cumulative_rewards)  # Sum of rewards across all episodes
        combined_trajectories += np.array(trajectories)
    
    learning_curves_dict[(alpha_actor, alpha_critic)] = combined_trajectories / n_runs
    
    # Return the negative cumulative reward to minimize
    mean_cumulative_reward = total_cumulative_reward / n_runs
    return -1 * mean_cumulative_reward  # We minimize the negative reward to maximize it



def discounted_reward(rewards, gamma):
    """
    Compute the discounted cumulative reward given a list of rewards for an episode.
    Args:
        rewards: List of rewards collected during an episode
        gamma: Discount factor
    """
    return sum([gamma**t * rewards[t] for t in range(len(rewards))])

def objective_discounted_reward(trial: optuna.Trial, 
                                ac_params: Dict[str, Any], 
                                learning_curves_dict: Dict[Tuple[float, float], np.ndarray], 
                                n_runs: int = 5) -> float:
    """
    Objective function that optimizes the discounted cumulative reward across multiple runs.
    """
    # Sample alpha_actor and alpha_critic 
    alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
    alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
    
    # Get actor-critic parameters
    nb_episodes = ac_params['nb_episodes']
    timeout = ac_params['timeout']
    gamma = ac_params['gamma']
    
    total_discounted_reward = 0
    combined_trajectories = np.zeros(nb_episodes)
    
    for _ in range(n_runs):
        env = create_maze_from_params(ac_params)
        _, _, trajectories, rewards_per_episode = naive_actor_critic_discounted_rewards(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
        
        # Compute discounted rewards for each episode
        discounted_rewards = [discounted_reward(rewards, gamma) for rewards in rewards_per_episode]
        total_discounted_reward += sum(discounted_rewards)  # Sum of discounted rewards across episodes
        combined_trajectories += np.array(trajectories)
    
    learning_curves_dict[(alpha_actor, alpha_critic)] = combined_trajectories / n_runs
    
    # Return the negative discounted reward to minimize
    mean_discounted_reward = total_discounted_reward / n_runs
    return -1 * mean_discounted_reward  # Minimize the negative discounted reward to maximize it



def objective_td_error(trial: optuna.Trial, 
                       ac_params: Dict[str, Any], 
                       learning_curves_dict: Dict[Tuple[float, float], np.ndarray], 
                       n_runs: int = 5) -> float:
    """
    Objective function that optimizes the mean temporal difference (TD) error across multiple runs.
    """
    # Sample alpha_actor and alpha_critic 
    alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
    alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
    
    nb_episodes = ac_params['nb_episodes']
    timeout = ac_params['timeout']
    gamma = ac_params['gamma']
    
    total_td_error = 0
    combined_trajectories = np.zeros(nb_episodes)
    
    for _ in range(n_runs):
        env = create_maze_from_params(ac_params)
        _, V, trajectories= naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
        
        # Calculate TD error (difference between consecutive values in V)
        td_errors = []
        for episode in range(len(V) - 1):
            td_error = V[episode + 1] - V[episode]  # Difference between consecutive values
            td_errors.append(abs(td_error))
        
        total_td_error += np.mean(td_errors)  # Mean TD error across all episodes
        combined_trajectories += np.array(trajectories)
    
    learning_curves_dict[(alpha_actor, alpha_critic)] = combined_trajectories / n_runs
    
    # Return the mean TD error (we want to minimize it)
    mean_td_error = total_td_error / n_runs
    return mean_td_error



def objective_policy_entropy(trial: optuna.Trial, 
                             ac_params: Dict[str, Any], 
                             learning_curves_dict: Dict[Tuple[float, float], np.ndarray], 
                             n_runs: int = 5) -> float:
    """
    Objective function that optimizes the mean policy entropy across multiple runs.
    """
    # Sample alpha_actor and alpha_critic 
    alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
    alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
    
    nb_episodes = ac_params['nb_episodes']
    timeout = ac_params['timeout']
    gamma = ac_params['gamma']
    
    total_entropy = 0
    combined_trajectories = np.zeros(nb_episodes)
    
    for _ in range(n_runs):
        env = create_maze_from_params(ac_params)
        _, _, trajectories, _, entropies = naive_actor_critic_dif(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
        
        total_entropy += np.mean(entropies)  # Mean entropy across all episodes
        combined_trajectories += np.array(trajectories)
    
    learning_curves_dict[(alpha_actor, alpha_critic)] = combined_trajectories / n_runs
    
    # Return the negative entropy (as we want to maximize it)
    mean_entropy = total_entropy / n_runs
    return mean_entropy  # We minimize the negative entropy to maximize exploration



from typing import Callable

def run_optimization(ac_params: Dict[str, Any], 
                     n_trials: int = 100, 
                     sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(),
                     objective_fn: Callable = objective  # Par défaut, on utilise cumulative reward
                     ) -> Tuple[optuna.Study, Dict[str, Any], float, List[Tuple[float, float]], 
                                List[float], List[float], Dict[Tuple[float, float], np.ndarray]]:
    """
    Runs hyperparameter optimization using Optuna with the given sampler and objective function.

    Args:
        ac_params: Dictionary containing actor-critic parameters.
        n_trials: Number of trials to perform for optimization.
        sampler: The Optuna sampler to use (defaults to TPESampler for Bayesian optimization).
        objective_fn: The objective function to use for optimization (e.g., cumulative reward, TD error, etc.).
    
    Returns:
        study: The Optuna study object containing the results.
        best_params: The best hyperparameters found.
        best_performance: The best performance (based on the chosen objective function).
        all_params: List of all evaluated hyperparameters (alpha_actor, alpha_critic).
        all_performances: Performance (based on the chosen objective function) for each set of hyperparameters.
        value_norms: The performance for each evaluated set of hyperparameters.
        learning_curves_dict: A dictionary mapping hyperparameter pairs to learning curves (steps per episode).
    """
    # Create a study object using the specified sampler
    study = optuna.create_study(direction='minimize', sampler=sampler)
    print(f"Sampler is {study.sampler.__class__.__name__}")
     
    learning_curves_dict: Dict[Tuple[float, float], np.ndarray] = {}  # Initialize a dictionary to store learning curves

    # Use partial to pass additional arguments to the chosen objective function
    objective_with_params = partial(objective_fn, ac_params=ac_params, learning_curves_dict=learning_curves_dict)

    # Optimize the chosen objective function for a given number of trials
    study.optimize(objective_with_params, n_trials=n_trials)

    # Extract results from the study
    best_params = study.best_params
    best_performance = study.best_value
    all_params = [(trial.params['alpha_actor'], trial.params['alpha_critic']) for trial in study.trials]
    all_performances = [trial.value for trial in study.trials]
    value_norms = all_performances

    # Return study results
    return study, best_params, best_performance, all_params, all_performances, value_norms, learning_curves_dict


def plot_heatmap_from_study(study):
    """
    Plots a heatmap from the study results showing the norm of the value function for different alpha_actor and alpha_critic values.

    Args:
        study: The Optuna study containing the optimization results.
    """
    # Extract the hyperparameters and their corresponding values from the study
    actor_values = []
    critic_values = []
    norm_values = []
    
    for trial in study.trials:
        actor_values.append(trial.params['alpha_actor'])
        critic_values.append(trial.params['alpha_critic'])
        norm_values.append(trial.value)
    
    # Create a scatter plot with a heatmap
    plt.figure(figsize=(10, 6))
    plt.scatter(actor_values, critic_values, c=norm_values, cmap='viridis', s=100)
    plt.colorbar(label='Norm of Value Function')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Alpha Actor')
    plt.ylabel('Alpha Critic')
    plt.title('Heatmap of Value Function Norms (Hyperparameters)')
    plt.show()


def plot_learning_curve(learning_curves_dict, best_params):
    """
    Plots the learning curve (steps per episode) for the best hyperparameters found by Optuna.

    Args:
        learning_curves_dict: Dictionary mapping hyperparameter pairs to learning curves (steps per episode).
        best_params: Best hyperparameters found by Optuna.
    """
    # Get the learning curve for the best hyperparameters
    best_alpha_actor = best_params['alpha_actor']
    best_alpha_critic = best_params['alpha_critic']
    best_learning_curve = learning_curves_dict[(best_alpha_actor, best_alpha_critic)]
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(best_learning_curve, label=f'Best Learning Curve (alpha_actor={best_alpha_actor}, alpha_critic={best_alpha_critic})')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps to Reach Goal')
    plt.title('Learning Curve for Best Hyperparameters')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    




def naive_actor_critic_discounted_rewards(
    env: MazeMDPEnv,
    alpha_actor: float,
    alpha_critic: float,
    gamma: float,
    nb_episodes: int = 1000,
    timeout: int = 50,
    render: bool = False,
):
    """
    Naive actor-critic algorithm.
    Returns:
        policy, value_function, trajectories, rewards_per_episode
    """
    # Initialize value function V(s) and the policy pi(a|s)
    V = np.zeros(env.nb_states)  # Critic (value function V(s))
    pi = np.ones((env.nb_states, env.action_space.n)) / env.action_space.n  # Actor (policy pi(a|s))

    # Function to renormalize the policy for a given state
    def renormalize_policy(pi, s):
        pi[s, :] = pi[s, :] / np.sum(pi[s, :])

    trajectories = []
    rewards_per_episode = []  # Store rewards for each episode

    for episode in range(nb_episodes):
        s, _ = env.reset(uniform=True)
        cpt = 0
        terminated = False
        truncated = False
        episode_rewards = []  # Store rewards for each step in this episode

        while not (terminated or truncated) and cpt < timeout:
            if render:
                env.draw_v_pi(V, pi.argmax(axis=1))

            # Sample an action from the current policy pi(a|s)
            a = sample_categorical(pi[s, :])

            # Perform a step in the environment
            s_next, r, terminated, truncated, _ = env.step(a)

            # Append reward for this step
            episode_rewards.append(r)

            # Calculate the temporal difference error (TD error)
            delta = r + gamma * V[s_next] * (1 - terminated) - V[s]

            # Update the critic (value function)
            V[s] = V[s] + alpha_critic * delta

            # Update the actor (policy) for the action taken
            pi_temp = pi[s, a] + alpha_actor * delta
            pi_temp = max(pi_temp, 1e-8)  # Ensure non-negative probability

            # Apply renormalization to ensure sum of probabilities is 1
            pi[s, a] = pi_temp
            renormalize_policy(pi, s)

            # Move to the next state
            s = s_next
            cpt += 1

        trajectories.append(cpt)
        rewards_per_episode.append(episode_rewards)  # Store all step rewards for this episode

    return pi, V, trajectories, rewards_per_episode  # Return the list of rewards per episode
