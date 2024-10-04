import optuna
import numpy as np
from functools import partial 
import numpy as np
from typing import List
import optuna
import gymnasium as gym
import matplotlib
from mazemdp.toolbox import sample_categorical
from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
from functools import partial
from typing import Dict, Tuple, Any
from typing import Callable
matplotlib.use("TkAgg")


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

####################################################################################################################################################################
####################################################################################################################################################################

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
    Naive actor-critic algorithm with entropy and tracking of V values over episodes.
    
    Args:
        env: The environment
        alpha_actor: Learning rate for the actor
        alpha_critic: Learning rate for the critic
        gamma: Discount factor
        nb_episodes: Number of episodes
        timeout: Maximum number of steps per episode
        render: Boolean flag to render the environment
    
    Returns:
        policy, V_per_episode, trajectories, entropies, rewards
    """
    V = np.zeros(env.nb_states)  # Critic (value function V(s))
    pi = np.ones((env.nb_states, env.action_space.n)) / env.action_space.n  # Actor (policy pi(a|s))

    def renormalize_policy(pi, s):
        pi[s, :] = pi[s, :] / np.sum(pi[s, :])

    trajectories = []
    entropies = []  # List to track entropy
    rewards_per_episode = []  # List to track total rewards for each episode
    delta_per_episode = []
    V_per_episode = []  # List to store V for each episode

    for episode in range(nb_episodes):
        s, _ = env.reset(uniform=True)
        cpt = 0
        terminated = False
        truncated = False
        entropy_episode = 0
        total_reward = 0  # Initialize total reward for this episode
        total_delta = 0  # Initialize total reward for this episode

        while not (terminated or truncated) and cpt < timeout:
            if render:
                env.draw_v_pi(V, pi.argmax(axis=1))

            a = sample_categorical(pi[s, :])
            s_next, r, terminated, truncated, _ = env.step(a)
            delta = r + gamma * V[s_next] * (1 - terminated) - V[s]
            V[s] = V[s] + alpha_critic * delta

            pi_temp = pi[s, a] + alpha_actor * delta
            pi_temp = max(pi_temp, 1e-8)
            pi[s, a] = pi_temp
            renormalize_policy(pi, s)

            # Calculate entropy for current state
            entropy_episode += -np.sum(pi[s, :] * np.log(pi[s, :] + 1e-8))

            # Accumulate reward
            total_reward += r
            total_delta += abs(delta)
            s = s_next
            cpt += 1

        # Store the value function V after each episode
        V_per_episode.append(V.copy())  # Use .copy() to avoid overwriting with future updates
        trajectories.append(cpt)  # Add the number of steps to reach the goal
        entropies.append(entropy_episode)  # Store entropy for this episode
        rewards_per_episode.append(total_reward)  # Store total reward for this episode
        delta_per_episode.append(total_delta)

    return pi, V_per_episode, trajectories, entropies, rewards_per_episode, delta_per_episode


####################################################################################################################################################################
####################################################################################################################################################################

def run_multiple_experiments(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, n_runs):
    """
    Runs the naive actor-critic algorithm multiple times and returns the results.

    Args:
        env: The environment.
        alpha_actor: Learning rate for the actor.
        alpha_critic: Learning rate for the critic.
        gamma: Discount factor.
        nb_episodes: Number of episodes for each run.
        timeout: Maximum number of steps per episode.
        n_runs: Number of independent runs.

    Returns:
        all_values: A list containing the value functions (V_per_episode) for each run.
        all_trajectories: A list where each element corresponds to the number of steps in each episode for a single run.
        all_entropies: A list where each element contains the entropy values for each episode in a single run.
        all_rewards: A list where each element contains the rewards collected for each episode in a single run.
    """
    all_values = []  # To store V_per_episode for each run
    all_trajectories = []
    all_entropies = []
    all_rewards = []  # To track the rewards for each run
    all_deltas = []
    for _ in range(n_runs):
        pi, V_per_episode, trajectories, entropies, rewards, deltas = naive_actor_critic(
            env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False
        )
        all_values.append(V_per_episode)  # Collect V_per_episode for this run
        all_trajectories.append(trajectories)
        all_entropies.append(entropies)
        all_rewards.append(rewards)  # Collect the rewards for this run
        all_deltas.append(deltas)

    return all_values, all_trajectories, all_entropies, all_rewards, all_deltas


####################################################################################################################################################################
####################################################################################################################################################################
class ActorCriticObjective:
    """
    Class defining multiple objective functions to optimize an actor-critic algorithm
    based on different criteria such as the norm of the value function, cumulative reward, 
    and discounted cumulative reward.
    """

    def __init__(self, ac_params: Dict[str, Any], n_runs: int = 5):
        """
        Initialize the ActorCriticObjective class with given parameters.

        Args:
            ac_params: Parameters for the actor-critic algorithm.
            n_runs: Number of independent runs to perform per trial.
        """
        self.ac_params = ac_params
        self.n_runs = n_runs

    def run_experiments(self, trial) -> Tuple:
        """
        Helper function to run multiple experiments with different configurations.
        """
        alpha_actor = trial.suggest_float('alpha_actor', 1e-1, 1.0)
        alpha_critic = trial.suggest_float('alpha_critic', 1e-1, 1.0)

        # Retrieve relevant parameters from ac_params
        nb_episodes = self.ac_params['nb_episodes']
        timeout = self.ac_params['timeout']
        gamma = self.ac_params['gamma']
        env = create_maze_from_params(self.ac_params)

        # Run experiments and return results
        all_values, all_trajectories, _, all_rewards, all_deltas= run_multiple_experiments(
            env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, self.n_runs
        )

        return all_values, all_trajectories, all_rewards, all_deltas
    
    def set_all_attributes(self, trial, cumulative_reward_mean, trajectories, combined_V_norm, deltas):
        """
        Set all trial attributes in one call.
        """
        trial.set_user_attr('cumulative_rewards', cumulative_reward_mean)
        trial.set_user_attr('trajectories', trajectories)
        trial.set_user_attr('value_norms', combined_V_norm)
        trial.set_user_attr('temporal_difference_error', deltas)

    def process_experiment_results(self, multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas):
        """
        Process and compute all the key metrics (value function norm, cumulative reward mean, trajectories)
        from multiple experiment runs.
        """
        # Optimize using the last V value for each run
        last_V_values = [V_per_episode[-1] for V_per_episode in multiple_values]
        combined_V = np.sum(last_V_values, axis=0)
        combined_V_norm = np.linalg.norm(combined_V)

        cumulative_rewards = [np.sum(rewards) for rewards in multiple_rewards]
        cumulative_reward_mean = np.mean(cumulative_rewards)
        trajectories = np.nanmean(multiple_trajectories, axis=0)
        
        multiple_deltas = np.mean(multiple_deltas, axis=0)
        return combined_V_norm, cumulative_reward_mean, trajectories, multiple_deltas
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def objective_value_function_norm(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize the norm of the value function.
        """
        multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas = self.run_experiments(trial)

        # Calculate results
        combined_V_norm, cumulative_reward_mean, trajectories, deltas = self.process_experiment_results(
            multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas
        )

        # Set trial attributes
        self.set_all_attributes(trial, cumulative_reward_mean, trajectories, combined_V_norm, deltas)

        return -combined_V_norm  # Maximize the norm of the value function

    def objective_cumulative_reward(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize cumulative rewards (without discounting).
        """
        multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas = self.run_experiments(trial)

        # Calculate results
        combined_V_norm, cumulative_reward_mean, trajectories, deltas = self.process_experiment_results(
            multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas
        )

        # Set trial attributes
        self.set_all_attributes(trial, cumulative_reward_mean, trajectories, combined_V_norm, deltas)

        return -cumulative_reward_mean  # Maximize the cumulative reward

    def objective_convergence_time(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize convergence time.
        """
        multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas = self.run_experiments(trial)

        # Calculate results
        combined_V_norm, cumulative_reward_mean, trajectories, deltas = self.process_experiment_results(
            multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas
        )

        # Set trial attributes
        self.set_all_attributes(trial, cumulative_reward_mean, trajectories, combined_V_norm, deltas)

        return np.mean(trajectories)  # Minimize the average number of steps
    
    def objective_discounted_cumulative_reward(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize discounted cumulative rewards.
        """
        multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas = self.run_experiments(trial)

        # Calculate results
        combined_V_norm, cumulative_reward_mean, trajectories, deltas = self.process_experiment_results(
            multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas
        )

        # Set trial attributes
        self.set_all_attributes(trial, cumulative_reward_mean, trajectories, combined_V_norm, deltas)

        # Calculate the discounted cumulative reward mean
        discounted_cumulative_rewards = []
        for rewards in multiple_rewards:
            discounted_reward = 0
            for t, reward in enumerate(rewards):
                discounted_reward += (self.ac_params['gamma'] ** t) * reward
            discounted_cumulative_rewards.append(discounted_reward)
        discounted_cumulative_reward_mean = np.mean(discounted_cumulative_rewards)

        return -discounted_cumulative_reward_mean  # Maximize the discounted cumulative reward

    def objective_deltas(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize the temporal difference error (deltas).
        """
        multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas = self.run_experiments(trial)

        # Calculate results
        combined_V_norm, cumulative_reward_mean, trajectories, deltas = self.process_experiment_results(
            multiple_values, multiple_trajectories, multiple_rewards, multiple_deltas
        )

        # Set trial attributes
        self.set_all_attributes(trial, cumulative_reward_mean, trajectories, combined_V_norm, deltas)

        return np.mean(deltas)  # Minimize the delta
    
    def select_objective(self, criterion: str) -> Callable:
        """
        Select the objective function based on the chosen criterion.

        Args:
            criterion: The criterion to optimize (e.g., 'value_norm', 'cumulative_reward').

        Returns:
            The corresponding objective function.
        """
        if criterion == "value_norm":
            return self.objective_value_function_norm
        elif criterion == "convergence_time":
            return self.objective_convergence_time
        elif criterion == "cumulative_reward":
            return self.objective_cumulative_reward
        elif criterion == "discounted_cumulative_reward":
            return self.objective_discounted_cumulative_reward
        elif criterion == "temporal_difference_error":
            return self.objective_deltas
        else:
            raise ValueError("Invalid criterion specified.")
        
####################################################################################################################################################################
####################################################################################################################################################################

def run_optimization(ac_params: Dict[str, Any], 
                     sampler: optuna.samplers.BaseSampler,
                     criterion: str = 'value_norm',  # The optimization criterion to use
                     n_trials: int = 100, 
                     n_runs: int = 5
                     ) -> Tuple[optuna.Study, Dict[str, Any], float, List[Tuple[float, float]], 
                                List[float], List[Any], List[Any], List[Any]]:
    """
    Runs hyperparameter optimization using Optuna for an actor-critic algorithm based on the specified objective function.
    
    This function executes multiple trials to find the best hyperparameters for the actor-critic algorithm using Optuna's
    sampler and a selected optimization criterion (e.g., value norm, cumulative rewards).
    
    Args:
        ac_params: A dictionary containing the actor-critic parameters, including:
            - 'nb_episodes': Number of episodes per trial.
            - 'timeout': Maximum steps per episode.
            - 'gamma': Discount factor for the rewards.
        sampler: The Optuna sampler to use for selecting hyperparameters (e.g., TPESampler for Bayesian optimization).
        criterion: The optimization criterion to use, which defines the objective function (e.g., 'value_norm', 'cumulative_reward').
        n_trials: The number of trials to run for hyperparameter optimization.
        n_runs: The number of independent runs per trial to average results and reduce variability.
    
    Returns:
        study: The Optuna Study object containing the complete results of the optimization process.
        best_params: A dictionary containing the best hyperparameters found during optimization (e.g., alpha_actor, alpha_critic).
        best_performance: The best performance score achieved according to the selected objective function.
        all_params: A list of tuples containing the evaluated hyperparameters (alpha_actor, alpha_critic) for each trial.
        value_norms: A list of value function norms (used when 'value_norm' is selected as the criterion) for each trial.
        trajectories: A list of average trajectory lengths (number of steps per episode) for each trial.
        cumulative_rewards: A list of cumulative rewards for each trial.
    """

    # Initialize the objective with criteria and base parameters
    objective_cls = ActorCriticObjective(ac_params=ac_params, n_runs=n_runs)

    # Validate the objective_cls parameter
    if not isinstance(objective_cls, ActorCriticObjective):
        raise ValueError("objective_cls must be an instance of ActorCriticObjective")

    # Get the objective function based on the criterion provided
    objective_fn = objective_cls.select_objective(criterion)

    # Create a study object using the specified sampler
    study = optuna.create_study(direction='minimize', sampler=sampler)
    print(f"Sampler is {study.sampler.__class__.__name__}")
    
    # Optimize the chosen objective function for a given number of trials
    study.optimize(objective_fn, n_trials=n_trials)

    # Extract results from the study
    best_params = study.best_params
    best_performance = study.best_value
    all_params = [(trial.params['alpha_actor'], trial.params['alpha_critic']) for trial in study.trials]
    value_norms = [trial.user_attrs['value_norms'] for trial in study.trials]
    trajectories = [trial.user_attrs['trajectories'] for trial in study.trials]
    cumulative_rewards = [trial.user_attrs['cumulative_rewards'] for trial in study.trials]
    deltas = [trial.user_attrs['temporal_difference_error'] for trial in study.trials]

    # Return study results
    return study, best_params, best_performance, all_params, value_norms, trajectories, cumulative_rewards, deltas