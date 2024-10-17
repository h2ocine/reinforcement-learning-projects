import torch 
import torch.nn as nn 
from torch.distributions import Normal
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params
from bbrl_utils.notebook import setup_tensorboard
from bbrl.visu.plot_policies import plot_policy
from omegaconf import OmegaConf
import bbrl_gymnasium
import math  
import copy 
from easypip import easyimport
mse = nn.MSELoss()

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        # Initialisation de l'agent Q. Il prend en entrée la dimension des états,
        # des couches cachées et la dimension des actions.
        super().__init__()
        self.is_q_function = True  # Indique qu'il s'agit d'une fonction Q.
        # Création du modèle MLP (Perceptron Multi-couches) avec les dimensions d'entrée (états + actions),
        # les couches cachées et une sortie scalaire (valeur Q).
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        # Récupération de l'état courant `s_t` et de l'action choisie `a_t`
        obs = self.get(("env/env_obs", t))  # Observation de l'environnement.
        action = self.get(("action", t))  # Action prise à ce temps.

        # Calcul de la valeur Q(s_t, a_t) en concaténant l'état et l'action.
        obs_act = torch.cat((obs, action), dim=1)  # Concaténation de l'état et de l'action.
        # Prédiction de la valeur Q et suppression de la dernière dimension (scalaire).
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)  # Enregistrement de la valeur Q.


class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        # Initialisation de l'agent acteur, qui prend en entrée les états et renvoie une action.
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        # Création du modèle MLP avec ReLU pour les couches cachées et Tanh pour les actions (bornées).
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t, **kwargs):
        # Récupération de l'état courant `s_t`
        obs = self.get(("env/env_obs", t))
        # Calcul de l'action déterministe.
        action = self.model(obs)
        # Enregistrement de l'action choisie.
        self.set(("action", t), action)


class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        # Initialisation de l'agent qui ajoute un bruit gaussien (bruit exploratoire).
        super().__init__()
        self.sigma = sigma  # Écart-type du bruit gaussien.

    def forward(self, t, **kwargs):
        # Récupération de l'action sans bruit.
        act = self.get(("action", t))
        # Ajout du bruit gaussien à l'action pour encourager l'exploration.
        dist = Normal(act, self.sigma)
        action = dist.sample()  # Génération de l'action bruitée.
        # Enregistrement de l'action bruitée.
        self.set(("action", t), action)


class AddOUNoise(Agent):
    """
    Ajoute un bruit via le processus d'Ornstein-Uhlenbeck pour les actions, recommandé dans DDPG.
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        # Initialisation des paramètres du processus OU.
        self.theta = theta  # Paramètre de retour à la moyenne.
        self.std_dev = std_dev  # Déviation standard du bruit.
        self.dt = dt  # Pas de temps.
        self.x_prev = 0  # Valeur précédente du bruit.

    def forward(self, t, **kwargs):
        # Récupération de l'action.
        act = self.get(("action", t))
        # Calcul du bruit OU à ajouter à l'action.
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x  # Mise à jour de la valeur du bruit.
        # Enregistrement de l'action avec le bruit OU ajouté.
        self.set(("action", t), x)


def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, target_q_values: torch.Tensor):
    """
    Calcul de la perte du critic dans DDPG. Utilise la perte quadratique moyenne (MSE).
    """
    # Calcul de la cible de la fonction de valeur Q : reward + valeur future actualisée.
    target = reward[1] + cfg.algorithm.discount_factor * target_q_values[1] * must_bootstrap[1].int()
    # Calcul de la perte MSE entre la valeur Q actuelle et la cible.
    return mse(q_values[0], target)


def compute_actor_loss(q_values):
    """
    Retourne la perte de l'acteur en DDPG : il essaie de maximiser Q(s, a).
    """
    return -q_values[0].mean()  # On maximise la valeur Q en minimisant son opposé.


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------- DDPG -----------------------------------------------
# ----------------------------------------------------------------------------------------------------
class DDPG(EpochBasedAlgo):
    def __init__(self, cfg, env_wrappers):
        super().__init__(cfg, env_wrappers)

        # Création du critic, target critic et actor, ainsi qu'un agent de bruit exploratoire.
        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
        self.critic = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic/")
        self.target_critic = copy.deepcopy(self.critic).with_prefix("target-critic/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # Ajout de bruit gaussien aux actions pour l'exploration.
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        # Politique d'entraînement avec exploration, et politique d'évaluation sans bruit.
        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        # Agents temporaires pour appliquer les modèles à travers le temps.
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic = TemporalAgent(self.critic)
        self.t_target_critic = TemporalAgent(self.target_critic)

        # Optimiseurs pour les réseaux.
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic)

def run_ddpg(ddpg: DDPG):
    # Listes pour enregistrer les données des courbes d'apprentissage
    critic_losses = []
    actor_losses = []
    rewards_per_step = []
    steps = []
    best_rewards = []
    episode_reward = 0  # Pour stocker la récompense totale d'un épisode
    best_reward = -float('inf')  # Variable pour suivre la meilleure récompense

    for rb in ddpg.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(ddpg.cfg.algorithm.batch_size)

        # Calcul de la perte du critic en utilisant la politique actuelle.
        ddpg.t_critic(rb_workspace, t=0, n_steps=1)  # Critic sur s, a
        with torch.no_grad():
            # Calcul de la politique de l'acteur et des valeurs Q pour la politique cible.
            ddpg.t_actor(rb_workspace, t=1, n_steps=1)  # Actor sur s'
            ddpg.t_target_critic(rb_workspace, t=1, n_steps=1)

        # Extraction des valeurs nécessaires (Q-values, récompenses, etc.).
        q_values, terminated, reward, target_q_values = rb_workspace[
            "critic/q_value", "env/terminated", "env/reward", "target-critic/q_value"
        ]
        must_bootstrap = ~terminated

        # Calcul de la récompense cumulée
        episode_reward += reward.mean().item()

        # Mise à jour du critic (backpropagation).
        critic_loss = compute_critic_loss(
            ddpg.cfg, reward, must_bootstrap, q_values, target_q_values
        )

        ddpg.critic_optimizer.zero_grad()
        critic_loss.backward()  # Calcul des gradients.
        torch.nn.utils.clip_grad_norm_(
            ddpg.critic.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.critic_optimizer.step()  # Mise à jour des poids du critic.

        # Calcul de la perte de l'acteur.
        ddpg.t_actor(rb_workspace, t=0, n_steps=1)  # Actor sur s
        ddpg.t_critic(rb_workspace, t=0, n_steps=1)  # Critic sur s, a

        q_values = rb_workspace["critic/q_value"]
        actor_loss = compute_actor_loss(q_values)

        # Mise à jour de l'acteur (backpropagation).
        ddpg.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.actor.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.actor_optimizer.step()

        # Mise à jour lente (soft update) du critic cible.
        soft_update_params(
            ddpg.critic, ddpg.target_critic, ddpg.cfg.algorithm.tau_target
        )

        # Enregistrement des valeurs pour les courbes d'apprentissage
        critic_losses.append(critic_loss.item())  # Sauvegarder la perte du critic
        actor_losses.append(actor_loss.item())  # Sauvegarder la perte de l'acteur
        rewards_per_step.append(reward.mean().item())  # Récompense moyenne par étape
        steps.append(ddpg.nb_steps)  # Ajouter le nombre total d'étapes

        # Mise à jour de la meilleure récompense
        if episode_reward > best_reward:
            best_reward = episode_reward
        best_rewards.append(best_reward)

        # Réinitialiser la récompense de l'épisode si l'environnement se termine
        if terminated.any():
            episode_reward = 0

    return critic_losses, actor_losses, rewards_per_step, steps, best_rewards


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------- TD3 -----------------------------------------------
# ----------------------------------------------------------------------------------------------------
class TD3(EpochBasedAlgo):
    def __init__(self, cfg, env_wrappers):
        super().__init__(cfg, env_wrappers)

        # Définition des agents pour TD3 : deux critics, un acteur et des critiques cibles.
        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic-1/")
        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic-2/")

        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic-1/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic-2/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # Ajout de bruit gaussien pour l'exploration.
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        # Agents pour appliquer les modèles sur plusieurs étapes.
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic_1 = TemporalAgent(self.critic_1)
        self.t_critic_2 = TemporalAgent(self.critic_2)
        self.t_target_critic_1 = TemporalAgent(self.target_critic_1)
        self.t_target_critic_2 = TemporalAgent(self.target_critic_2)

        # Optimiseurs pour les réseaux.
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1, self.critic_2)

def run_td3(td3: TD3):
    # Listes pour enregistrer les données des courbes d'apprentissage
    critic_losses = []
    actor_losses = []
    rewards_per_step = []
    steps = []
    best_rewards = []

    # Boucle principale d'apprentissage TD3, qui itère sur le replay buffer (mémoire d'expériences)
    for rb in td3.iter_replay_buffers():
        # Extraction d'un batch d'expériences aléatoires depuis le replay buffer
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)

        # Mise à jour des réseaux de critiques à t=0
        td3.t_critic_1(rb_workspace, t=0, n_steps=1)  # Calcul des valeurs Q pour critic_1
        td3.t_critic_2(rb_workspace, t=0, n_steps=1)  # Calcul des valeurs Q pour critic_2
        
        # Avec torch.no_grad() pour désactiver la backpropagation (pas de mise à jour ici)
        with torch.no_grad():
            # Application de l'acteur pour générer les actions futures à t=1
            td3.t_actor(rb_workspace, t=1, n_steps=1)
            # Calcul des valeurs Q cibles à t=1
            td3.t_target_critic_1(rb_workspace, t=1, n_steps=1)
            td3.t_target_critic_2(rb_workspace, t=1, n_steps=1)

        # Récupération des informations nécessaires du workspace pour les mises à jour
        q_values_1, q_values_2, terminated, reward, target_q_values_1, target_q_values_2 = rb_workspace[
            "critic-1/q_value", "critic-2/q_value", "env/terminated", "env/reward", "target-critic-1/q_value", "target-critic-2/q_value"
        ]
        # Calcul du masque pour identifier les transitions à bootstrapper (non terminées)
        must_bootstrap = ~terminated

        # Calcul des valeurs Q cibles : on prend le minimum des valeurs Q des deux critiques cibles (principe clé de TD3 pour éviter la surestimation des valeurs Q)
        target_q_values = torch.min(target_q_values_1, target_q_values_2)

        # --- Mise à jour des critiques ---
        # Calcul de la perte pour critic_1 en utilisant la fonction compute_critic_loss
        critic_loss_1 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_1, target_q_values
        )
        # Calcul de la perte pour critic_2
        critic_loss_2 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_2, target_q_values
        )
        # La perte totale des critiques est la somme des deux
        critic_loss = critic_loss_1 + critic_loss_2

        # Gradient step (critic)
        td3.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer.step()

        # --- Mise à jour de l'acteur ---
        # Calcul des actions en utilisant l'acteur à t=0
        td3.t_actor(rb_workspace, t=0, n_steps=1)
        # Mise à jour du critic_1 avec ces actions
        td3.t_critic_1(rb_workspace, t=0, n_steps=1)

        # Récupération des valeurs Q issues de critic_1
        q_values = rb_workspace["critic-1/q_value"]
        # Calcul de la perte de l'acteur : l'objectif est de maximiser les valeurs Q sous la politique actuelle
        actor_loss = compute_actor_loss(q_values)
        
        # Gradient step pour l'acteur
        td3.actor_optimizer.zero_grad()  # Réinitialisation des gradients
        actor_loss.backward()  # Calcul des gradients via backpropagation
        # Clipping de gradient pour l'acteur
        torch.nn.utils.clip_grad_norm_(
            td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        # Application de la mise à jour de l'acteur
        td3.actor_optimizer.step()

        # --- Mise à jour des cibles critiques avec soft update ---
        # Application de la mise à jour douce (soft update) des paramètres de critic_1 et target_critic_1
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )
        # Application de la mise à jour douce pour critic_2 et target_critic_2
        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )

        
        # Sauvegarde des statistiques d'apprentissage à retrouver dans les logs TensorBoard
        if ((td3.nb_steps - td3.last_eval_step) > td3.cfg.algorithm.eval_interval):
            td3.evaluate() # Evaluation et log built-in dans BBRL
            td3.logger.add_log("actor_loss", actor_loss, td3.nb_steps)
            td3.logger.add_log("critic_loss", critic_loss, td3.nb_steps)
            td3.logger.add_log("reward_per_episode", reward.mean(), td3.nb_steps)

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            rewards_per_step.append(reward.mean())
            steps.append(td3.nb_steps)
            best_rewards.append(td3.best_reward) # À appeler après td3.evaluate() pour bien la mettre à jour

    return critic_losses, actor_losses, rewards_per_step, steps, best_rewards