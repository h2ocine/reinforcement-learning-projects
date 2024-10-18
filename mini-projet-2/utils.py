import matplotlib.pyplot as plt

def plot_metrics(critic_losses, actor_losses, running_rewards):
    """
    Fonction pour tracer les critic loss, actor loss et running rewards.
    
    Args:
        critic_losses (list or tensor): Liste ou tenseur des pertes du critic.
        actor_losses (list or tensor): Liste ou tenseur des pertes de l'acteur.
        running_rewards (list or tensor): Liste ou tenseur des running rewards.
    """
    plt.figure(figsize=(7, 8))
    
    # Plot Critic Loss
    plt.subplot(3, 1, 1)
    plt.plot(critic_losses, label="Critic Loss", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Critic Loss per Episode")
    plt.legend()
    plt.grid(True)
    
    # Plot Actor Loss
    plt.subplot(3, 1, 2)
    plt.plot(actor_losses, label="Actor Loss", color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Actor Loss per Episode")
    plt.legend()
    plt.grid(True)
    
    # Plot Running Rewards (au milieu)
    plt.subplot(3, 1, 3)
    plt.plot(running_rewards, label="Running Rewards", color='orange')
    plt.xlabel("Episodes")
    plt.ylabel("Running Rewards")
    plt.title("Running Rewards per Episode")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


import os
import json

def plot_results(results, save_path=None):
    """
    Fonction pour tracer et sauvegarder les résultats des différents algorithmes et environnements.
    
    Args:
        results (dict): Résultats d'entraînement pour différents algorithmes et environnements.
        save_path (str): Chemin où sauvegarder les graphiques. Si None, n'enregistre pas.
    """
    # Crée le répertoire de sauvegarde si nécessaire
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    for key, value in results.items():
        critic_losses = value['critic_losses']
        actor_losses = value['actor_losses']
        running_rewards = value['running_rewwards']
        
        print(f"Plotting results for {key}")
        plt.figure(figsize=(7, 8))
        
        # Plot Critic Loss
        plt.subplot(3, 1, 1)
        plt.plot(critic_losses, label="Critic Loss", color='blue')
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title(f"{key} - Critic Loss per Episode")
        plt.legend()
        plt.grid(True)
        
        # Plot Actor Loss
        plt.subplot(3, 1, 2)
        plt.plot(actor_losses, label="Actor Loss", color='green')
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title(f"{key} - Actor Loss per Episode")
        plt.legend()
        plt.grid(True)
        
        # Plot Running Rewards
        plt.subplot(3, 1, 3)
        plt.plot(running_rewards, label="Running Rewards", color='orange')
        plt.xlabel("Episodes")
        plt.ylabel("Running Rewards")
        plt.title(f"{key} - Running Rewards per Episode")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarde du plot si un chemin est fourni
        if save_path:
            file_name = os.path.join(save_path, f"{key}_plot.png")
            plt.savefig(file_name)
            print(f"Saved plot for {key} at {file_name}")
        
        # Affichage du plot
        plt.show()


def save_results_to_json(results, save_path='results.json'):
    """
    Enregistre les résultats dans un fichier JSON.
    
    Args:
        results (dict): Résultats à enregistrer.
        save_path (str): Chemin où enregistrer le fichier JSON.
    """
    with open(save_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {save_path}")