import matplotlib.pyplot as plt

def plot_learning_curves(results):
    """
    Tracer les courbes d'apprentissage pour tous les environnements et algorithmes dans 'results'.
    Le dictionnaire 'results' doit contenir des données structurées pour chaque environnement et chaque algorithme.
    
    :param results: Dictionnaire contenant les statistiques d'apprentissage pour chaque environnement et algorithme.
    """
    for key, stats in results.items():
        critic_losses = stats["critic_losses"]
        actor_losses = stats["actor_losses"]
        rewards_per_step = stats["rewards_per_step"]
        steps = stats["steps"]
        best_rewards = stats["best_rewards"]

        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Learning Curves for {key}', fontsize=16)

        # Tracer la perte des critiques
        plt.subplot(2, 2, 1)
        plt.plot(steps, critic_losses, label=f'Critic Losses ({key})')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Critic Losses")
        plt.legend()

        # Tracer la perte de l'acteur
        plt.subplot(2, 2, 2)
        plt.plot(steps, actor_losses, label=f'Actor Losses ({key})')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Actor Losses")
        plt.legend()

        # Tracer les récompenses par étape
        plt.subplot(2, 2, 3)
        plt.plot(steps, rewards_per_step, label=f'Rewards per Step ({key})')
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Rewards per Step")
        plt.legend()

        # Tracer les meilleures récompenses
        plt.subplot(2, 2, 4)
        plt.plot(steps, best_rewards, label=f'Best Rewards ({key})')
        plt.xlabel("Steps")
        plt.ylabel("Best Reward")
        plt.title("Best Rewards")
        plt.legend()

        plt.tight_layout()
        plt.show()
