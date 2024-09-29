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
    Naive actor-critic algorithm with entropy and convergence tracking.
    Args:
        env: The environment
        alpha_actor: Learning rate for the actor
        alpha_critic: Learning rate for the critic
        gamma: Discount factor
        nb_episodes: Number of episodes
        timeout: Maximum number of steps per episode
        render: Boolean flag to render the environment
    Returns:
        policy, value_function, trajectories, entropies, convergence_steps
    """
    V = np.zeros(env.nb_states)  # Critic (value function V(s))
    pi = np.ones((env.nb_states, env.action_space.n)) / env.action_space.n  # Actor (policy pi(a|s))
    convergence=False

    def renormalize_policy(pi, s):
        pi[s, :] = pi[s, :] / np.sum(pi[s, :])

    trajectories = []
    entropies = []  # List to track entropy
    convergence_steps = 0

    for episode in range(nb_episodes):
        s, _ = env.reset(uniform=True)
        cpt = 0
        terminated = False
        truncated = False
        entropy_episode = 0

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

            s = s_next
            cpt += 1

        trajectories.append(cpt)
        entropies.append(entropy_episode)  # Store entropy for this episode

        # Check for convergence (e.g., policy does not change significantly)
        if episode > 1 and np.allclose(pi, previous_pi, atol=1e-3) and not convergence:
            convergence_steps = episode

        previous_pi = pi.copy()

    return pi, V, trajectories, entropies, convergence_steps

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
        all_trajectories: A list where each element corresponds to the number of steps in each episode for a single run.
        all_entropies: A list where each element contains the entropy values for each episode in a single run.
        all_convergence_steps: A list containing the convergence step for each run.
    """
    all_trajectories = []
    all_entropies = []
    all_convergence_steps = []

    for _ in range(n_runs):
        pi, V, trajectories, entropies, convergence_steps = naive_actor_critic(
            env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False
        )
        all_trajectories.append(trajectories)
        all_entropies.append(entropies)
        all_convergence_steps.append(convergence_steps)

    return all_trajectories, all_entropies, all_convergence_steps


####################################################################################################################################################################
####################################################################################################################################################################
class ActorCriticObjective:
    """
    Class defining multiple objective functions to optimize an actor-critic algorithm
    based on different criteria such as the norm of the value function, convergence time, 
    average number of steps to convergence, cumulative reward, and discounted cumulative reward.

    Available Criteria:
    1. Convergence Time: The number of episodes required for the algorithm to converge.
    2. Value Function Norm: Measures long-term policy performance in terms of state evaluations.
    3. Average Steps to Convergence: Measures the number of actions taken before the algorithm converges.
    4. Cumulative Reward: Evaluates the cumulative rewards over time.
    5. Discounted Cumulative Reward: Evaluates the cumulative rewards over time, weighted by the gamma factor to favor immediate rewards.
    """

    def __init__(self, ac_params: Dict[str, Any], n_runs: int = 5):
        """
        Class initialization.

        Args:
            ac_params: Parameters of the actor-critic algorithm.
            n_runs: Number of independent runs to perform for each configuration.
        """
        self.ac_params = ac_params
        self.n_runs = n_runs

    def objective_value_function_norm(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize the norm of the value function.

        Args:
            trial: Optuna trial object to suggest hyperparameters.

        Returns:
            mean_value_norm: The average norm of the value function across multiple runs.
        """
        alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
        alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
        
        nb_episodes = self.ac_params['nb_episodes']
        timeout = self.ac_params['timeout']
        gamma = self.ac_params['gamma']
        
        total_value_norm = 0
        
        for _ in range(self.n_runs):
            env = create_maze_from_params(self.ac_params)
            _, V, _ = naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
            total_value_norm += np.linalg.norm(V)
        
        mean_value_norm = total_value_norm / self.n_runs
        return -mean_value_norm  # Minimize the norm

    def objective_convergence_time(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize convergence time (number of episodes to converge).

        Args:
            trial: Optuna trial object to suggest hyperparameters.

        Returns:
            mean_convergence_steps: The average number of episodes to reach convergence.
        """
        alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
        alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
        
        nb_episodes = self.ac_params['nb_episodes']
        timeout = self.ac_params['timeout']
        gamma = self.ac_params['gamma']
        
        total_convergence_steps = 0
        
        for _ in range(self.n_runs):
            env = create_maze_from_params(self.ac_params)
            _, _, _, _, convergence_steps = naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
            total_convergence_steps += convergence_steps
        
        mean_convergence_steps = total_convergence_steps / self.n_runs
        return mean_convergence_steps  # Minimize the number of steps to convergence

    def objective_steps_to_convergence(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize the average number of steps per episode to reach convergence.

        Args:
            trial: Optuna trial object to suggest hyperparameters.

        Returns:
            mean_steps: The average number of steps per episode across multiple runs.
        """
        alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
        alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
        
        nb_episodes = self.ac_params['nb_episodes']
        timeout = self.ac_params['timeout']
        gamma = self.ac_params['gamma']
        
        total_steps = 0
        
        for _ in range(self.n_runs):
            env = create_maze_from_params(self.ac_params)
            _, _, trajectories, _, _ = naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
            total_steps += np.mean(trajectories)
        
        mean_steps = total_steps / self.n_runs
        return mean_steps  # Minimize the average number of steps

    def objective_cumulative_reward(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize cumulative rewards (without discounting).

        Args:
            trial: Optuna trial object to suggest hyperparameters.

        Returns:
            mean_cumulative_reward: The average cumulative reward across multiple runs.
        """
        alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
        alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
        
        nb_episodes = self.ac_params['nb_episodes']
        timeout = self.ac_params['timeout']
        gamma = self.ac_params['gamma']
        
        total_cumulative_reward = 0
        
        for _ in range(self.n_runs):
            env = create_maze_from_params(self.ac_params)
            _, _, trajectories, _, _ = naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
            
            cumulative_reward = np.sum(trajectories)  # Sum of rewards without discounting
            total_cumulative_reward += cumulative_reward
        
        mean_cumulative_reward = total_cumulative_reward / self.n_runs
        return mean_cumulative_reward  # Maximize cumulative reward

    def objective_discounted_cumulative_reward(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize discounted cumulative rewards.

        Args:
            trial: Optuna trial object to suggest hyperparameters.

        Returns:
            mean_discounted_cumulative_reward: The average discounted cumulative reward across multiple runs.
        """
        alpha_actor = trial.suggest_float('alpha_actor', 1e-5, 1.0, log=True)
        alpha_critic = trial.suggest_float('alpha_critic', 1e-5, 1.0, log=True)
        
        nb_episodes = self.ac_params['nb_episodes']
        timeout = self.ac_params['timeout']
        gamma = self.ac_params['gamma']
        
        total_discounted_cumulative_reward = 0
        
        for _ in range(self.n_runs):
            env = create_maze_from_params(self.ac_params)
            _, _, trajectories, _, _ = naive_actor_critic(env, alpha_actor, alpha_critic, gamma, nb_episodes, timeout, render=False)
            
            # Calculate discounted reward
            discounted_cumulative_reward = np.sum([gamma**t * r for t, r in enumerate(trajectories)])
            total_discounted_cumulative_reward += discounted_cumulative_reward
        
        mean_discounted_cumulative_reward = total_discounted_cumulative_reward / self.n_runs
        return mean_discounted_cumulative_reward  # Maximize discounted cumulative reward

    def select_objective(self, criterion: str) -> Callable:
        """
        Select the objective function based on the chosen criterion.

        Args:
            criterion: The performance criterion ('value_norm', 'convergence_time', 'steps_to_convergence', 'cumulative_reward', 'discounted_cumulative_reward').

        Returns:
            Callable: The corresponding objective function based on the criterion.
        """
        if criterion == "value_norm":
            return self.objective_value_function_norm
        elif criterion == "convergence_time":
            return self.objective_convergence_time
        elif criterion == "steps_to_convergence":
            return self.objective_steps_to_convergence
        elif criterion == "cumulative_reward":
            return self.objective_cumulative_reward
        elif criterion == "discounted_cumulative_reward":
            return self.objective_discounted_cumulative_reward
        else:
            raise ValueError("Invalid criterion specified.")

####################################################################################################################################################################
####################################################################################################################################################################

def run_optimization(ac_params: Dict[str, Any], 
                     n_trials: int = 100, 
                     sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(),
                     objective_cls: ActorCriticObjective = None,  # The class containing the objective functions
                     criterion: str = 'cumulative_reward'  # The optimization criterion to use
                     ) -> Tuple[optuna.Study, Dict[str, Any], float, List[Tuple[float, float]], 
                                List[float], List[float]]:
    """
    Runs hyperparameter optimization using Optuna with the given sampler and objective function.

    Args:
        ac_params: Dictionary containing actor-critic parameters.
        n_trials: Number of trials to perform for optimization.
        sampler: The Optuna sampler to use (defaults to TPESampler for Bayesian optimization).
        objective_cls: Instance of ActorCriticObjective, which contains different objective functions.
        criterion: The optimization criterion to use (e.g., 'value_norm', 'convergence_time', 
                   'steps_to_convergence', 'cumulative_reward', 'discounted_cumulative_reward').
    
    Returns:
        study: The Optuna study object containing the results.
        best_params: The best hyperparameters found.
        best_performance: The best performance (based on the chosen objective function).
        all_params: List of all evaluated hyperparameters (alpha_actor, alpha_critic).
        all_performances: Performance (based on the chosen objective function) for each set of hyperparameters.
        value_norms: The performance for each evaluated set of hyperparameters.
    """
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
    all_performances = [trial.value for trial in study.trials]
    value_norms = all_performances  # You can rename this depending on what the criterion is optimizing

    # Return study results
    return study, best_params, best_performance, all_params, all_performances, value_norms