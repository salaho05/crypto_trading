# Crypto Trading Bot with Reinforcement Learning

Un projet complet de trading automatisé utilisant l'apprentissage par renforcement (RL) pour trader les cryptomonnaies.

## Description

Ce projet développe un agent de trading basé sur le reinforcement learning capable de:
- Analyser les données du marché des cryptomonnaies en temps réel
- Prendre des décisions d'achat/vente/hold via un agent RL entraîné
- Backtester les stratégies sur des données historiques
- Optimiser les hyperparamètres via Optuna
- Exécuter du trading en direct sur les échanges (CCXT)

## Structure du Projet

```
├── data/              # Données historiques et en temps réel
├── env/               # Environnements Gymnasium personnalisés
├── agents/            # Modèles RL (PPO, DQN, etc.)
├── backtesting/       # Module de backtesting
├── optimization/      # Optimisation des hyperparamètres
├── models/            # Modèles entraînés et sauvegardés
├── analysis/          # Analyse et visualisation des résultats
├── live_trading/      # Exécution en direct
├── config/            # Fichiers de configuration
└── main.py            # Point d'entrée principal
```

## Installation

1. Cloner le repository
```bash
git clone <repo-url>
cd crypto_trading_bot
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Dépendances Principales

- **ccxt**: API pour accéder aux échanges de cryptomonnaies
- **pandas, numpy**: Manipulation et analyse de données
- **ta**: Indicateurs techniques
- **gymnasium**: Framework pour créer des environnements RL
- **stable-baselines3**: Implémentations d'algorithmes RL
- **torch**: Deep learning
- **optuna**: Optimisation bayésienne des hyperparamètres
- **matplotlib, seaborn**: Visualisation

## Utilisation

```bash
# Entraîner un modèle
python main.py --train

# Backtester une stratégie
python main.py --backtest

# Optimiser les hyperparamètres
python main.py --optimize

# Lancer le trading en direct
python main.py --live
```

## Configuration

Les paramètres de configuration sont gérés dans le dossier `config/`. Voir le fichier de config pour les détails.

## License

MIT
