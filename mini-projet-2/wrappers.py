import numpy as np
import gymnasium as gym

class FeatureFilterWrapper(gym.Wrapper):
    def __init__(self, env, feature_idx):
        # Initialisation du wrapper avec l'environnement et l'indice de la caractéristique à masquer.
        super(FeatureFilterWrapper, self).__init__(env)
        self.feature_idx = feature_idx
        
        # Adapter l'espace d'observation en retirant une dimension.
        low = np.delete(self.observation_space.low, feature_idx)
        high = np.delete(self.observation_space.high, feature_idx)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed = None, options = None):
        """
        reset environment to initial state and remove the designated feature
        :param kwargs: keyword arguments for the reset function
        :return: filtered observation, info
        """
        obs, info = self.env.reset()  # get initial state
        filtered_obs = np.delete(obs, self.feature_idx)  # delete designated features
        return filtered_obs, info

    def step(self, action):
        # Exécute une action et masque la caractéristique dans l'observation retournée.
        obs, reward, done, truncated, info = self.env.step(action)
        filtered_obs = np.delete(obs, self.feature_idx)
        return filtered_obs, reward, done, truncated, info
    

class ObsTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, memory_size=1):
        # Initialisation du wrapper avec une mémoire des observations.
        super(ObsTimeExtensionWrapper, self).__init__(env)
        self.memory_size = memory_size
        self.prev_obs = np.zeros_like(self.env.observation_space.low)
        
        # Étendre l'espace d'observation en incluant les observations passées.
        low = np.concatenate([self.observation_space.low] * (memory_size + 1))
        high = np.concatenate([self.observation_space.high] * (memory_size + 1))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Réinitialise l'environnement et la mémoire des observations.
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_obs = np.zeros_like(obs)  # Réinitialisation de la mémoire.
        extended_obs = np.concatenate([self.prev_obs, obs])
        return extended_obs, info

    def step(self, action):
        # Exécute une action et combine l'observation actuelle avec la précédente.
        result = self.env.step(action)

        # Gestion du retour avec 4 ou 5 éléments.
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        extended_obs = np.concatenate([self.prev_obs, obs])
        self.prev_obs = obs  # Met à jour la mémoire avec l'observation courante.
        return extended_obs, reward, done, truncated, info
    
class ActionTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=1):
        # Initialisation avec l'environnement et la répétition d'actions.
        super(ActionTimeExtensionWrapper, self).__init__(env)
        self.action_repeat = action_repeat
        
        # Étendre l'espace d'action pour gérer les séquences d'actions.
        low = np.concatenate([self.action_space.low] * action_repeat)
        high = np.concatenate([self.action_space.high] * action_repeat)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        # Exécuter uniquement la première action de la séquence d'actions.
        first_action = action[:self.action_space.shape[0] // self.action_repeat]
        obs, reward, done, truncated, info = self.env.step(first_action)
        return obs, reward, done, truncated, info