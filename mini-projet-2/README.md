# Projet : Etude des Algorithmes TD3 et DDPG avec des Wrappers dans des Environnements Partiellement Observables

## Description du projet

Ce projet a pour objectif d'étudier l'impact des environnements partiellement observables sur les algorithmes de reinforcement learning TD3 (Twin Delayed DDPG) et DDPG (Deep Deterministic Policy Gradient). Nous avons implémenté deux types de wrappers — \textit{ActionTimeExtensionWrapper} et \textit{ObsTimeExtensionWrapper} — afin d'évaluer leurs impacts respectifs sur l'amélioration des performances dans ces environnements. Le projet se concentre sur l'entraînement des agents avec différents niveaux d'observabilité et sur l'analyse des résultats via les \textit{rewards} et les courbes de performances.

## Structure du projet

Le projet se divise en plusieurs fichiers et blocs de code :

- **train.ipynb** : Le notebook principal contenant les étapes d'entraînement pour TD3 et DDPG dans des environnements totalement et partiellement observables. C'est la partie la plus importante du code, où l'on applique les wrappers pour les différents scénarios d'observabilité. Les tests incluent des environnements où certaines variables d'état (comme \(\dot{x}\) et \(\theta\)) sont occultées, ainsi que des combinaisons avec les wrappers pour mesurer leur impact.
  
  - *Training complet*: L'algorithme s'entraîne à la fois dans un environnement complètement observable et dans des environnements partiellement observables (en retirant \(\dot{x}\), \(\theta\), ou les deux).
  
  - *Application des Wrappers*: Le notebook applique le \textit{ActionTimeExtensionWrapper}, le \textit{ObsTimeExtensionWrapper} et leur combinaison sur les environnements d'entraînement.

- **wrappers.py** : Ce fichier contient l'implémentation des trois wrappers utilisés dans le projet :
  - `ActionTimeExtensionWrapper` : Répète les actions dans le temps pour capturer une meilleure dynamique.
  - `ObsTimeExtensionWrapper` : Étend l'observation pour inclure un historique d'observations passées.
  - `FeatureFilterWrapper` : Masque certaines variables d'état pour rendre l'environnement partiellement observable.

- **td3_ddpg.py** : Ce fichier implémente les algorithmes TD3 et DDPG. Il contient les principales fonctions de ces algorithmes, y compris la gestion des replay buffers, la mise à jour des réseaux de l'acteur et du critique, ainsi que l'entraînement en utilisant les hyperparamètres définis.
