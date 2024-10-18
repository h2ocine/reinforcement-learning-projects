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
        obs, info = self.env.reset(seed=seed,options=options)  # get initial state
        filtered_obs = np.delete(obs, self.feature_idx)  # delete designated features
        return filtered_obs, info

    def step(self, action):
        # Exécute une action et masque la caractéristique dans l'observation retournée.
        obs, reward, done, truncated, info = self.env.step(action)
        filtered_obs = np.delete(obs, self.feature_idx)
        return filtered_obs, reward, done, truncated, info
    

class ObsTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, memory_size=1):
        # Initialisation avec une mémoire des observations plus longue.
        super(ObsTimeExtensionWrapper, self).__init__(env)
        self.memory_size = memory_size
        self.prev_obs = [np.zeros_like(self.env.observation_space.low) for _ in range(memory_size)]

        # Étendre l'espace d'observation en incluant les observations passées.
        low = np.concatenate([self.observation_space.low] * (memory_size + 1))
        high = np.concatenate([self.observation_space.high] * (memory_size + 1))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Réinitialiser l'environnement et la mémoire des observations.
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_obs = [np.zeros_like(obs) for _ in range(self.memory_size)]
        extended_obs = np.concatenate(self.prev_obs + [obs])
        return extended_obs, info

    def step(self, action):
        # Exécuter une action et combiner l'observation actuelle avec toutes les précédentes.
        obs, reward, done, truncated, info = self.env.step(action)
        self.prev_obs = self.prev_obs[1:] + [obs]  # Mettre à jour la mémoire.
        extended_obs = np.concatenate(self.prev_obs + [obs])
        return extended_obs, reward, done, truncated, info


class ActionTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=1):
        # Initialisation avec un nombre configurable de répétitions d'actions.
        super(ActionTimeExtensionWrapper, self).__init__(env)
        self.action_repeat = action_repeat
        
        # Étendre l'espace d'action pour gérer plus de répétitions d'actions.
        low = np.concatenate([self.action_space.low] * action_repeat)
        high = np.concatenate([self.action_space.high] * action_repeat)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        # Exécuter une séquence d'actions répétées.
        obs, total_reward, done, truncated, info = None, 0, False, False, {}
        
        for i in range(self.action_repeat):
            if done:
                break
            obs, reward, done, truncated, info = self.env.step(action[i])
            total_reward += reward  # Accumuler les récompenses sur plusieurs étapes.

        return obs, total_reward, done, truncated, info



