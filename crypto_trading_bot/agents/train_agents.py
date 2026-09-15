import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from env.crypto_trading_env import CryptoTradingEnv
import time
from multiprocessing import Process, Queue
import warnings

warnings.filterwarnings('ignore')

# Créer le répertoire models s'il n'existe pas
os.makedirs('agents/models', exist_ok=True)

def train_dqn(df, results_queue):
    """Entraîne un agent DQN"""
    print("\n" + "="*60)
    print("🤖 Entraînement DQN")
    print("="*60)

    start_time = time.time()

    try:
        env = CryptoTradingEnv(df, initial_balance=10000)
        env = DummyVecEnv([lambda: env])

        agent = DQN(
            'MlpPolicy',
            env,
            learning_rate=0.0001,
            verbose=0,
            buffer_size=10000,
            exploration_fraction=0.1,
            device='cpu'
        )

        print("Training DQN... (50000 timesteps)")
        agent.learn(total_timesteps=50000)
        agent.save('agents/models/dqn_model')
        print("✅ DQN model saved")

        elapsed_time = time.time() - start_time

        # Évaluer
        mean_return = evaluate_agent(agent, df, n_episodes=5)

        results_queue.put({
            'name': 'DQN',
            'return': mean_return,
            'time': elapsed_time,
            'agent': agent
        })

    except Exception as e:
        print(f"❌ Erreur DQN: {str(e)}")
        results_queue.put({'name': 'DQN', 'error': str(e)})

def train_ppo(df, results_queue):
    """Entraîne un agent PPO"""
    print("\n" + "="*60)
    print("🤖 Entraînement PPO")
    print("="*60)

    start_time = time.time()

    try:
        env = CryptoTradingEnv(df, initial_balance=10000)
        env = DummyVecEnv([lambda: env])

        agent = PPO(
            'MlpPolicy',
            env,
            learning_rate=0.0003,
            verbose=0,
            n_steps=2048,
            device='cpu'
        )

        print("Training PPO... (50000 timesteps)")
        agent.learn(total_timesteps=50000)
        agent.save('agents/models/ppo_model')
        print("✅ PPO model saved")

        elapsed_time = time.time() - start_time

        # Évaluer
        mean_return = evaluate_agent(agent, df, n_episodes=5)

        results_queue.put({
            'name': 'PPO',
            'return': mean_return,
            'time': elapsed_time,
            'agent': agent
        })

    except Exception as e:
        print(f"❌ Erreur PPO: {str(e)}")
        results_queue.put({'name': 'PPO', 'error': str(e)})

def train_a2c(df, results_queue):
    """Entraîne un agent A2C"""
    print("\n" + "="*60)
    print("🤖 Entraînement A2C")
    print("="*60)

    start_time = time.time()

    try:
        env = CryptoTradingEnv(df, initial_balance=10000)
        env = DummyVecEnv([lambda: env])

        agent = A2C(
            'MlpPolicy',
            env,
            learning_rate=0.0007,
            verbose=0,
            device='cpu'
        )

        print("Training A2C... (50000 timesteps)")
        agent.learn(total_timesteps=50000)
        agent.save('agents/models/a2c_model')
        print("✅ A2C model saved")

        elapsed_time = time.time() - start_time

        # Évaluer
        mean_return = evaluate_agent(agent, df, n_episodes=5)

        results_queue.put({
            'name': 'A2C',
            'return': mean_return,
            'time': elapsed_time,
            'agent': agent
        })

    except Exception as e:
        print(f"❌ Erreur A2C: {str(e)}")
        results_queue.put({'name': 'A2C', 'error': str(e)})

def evaluate_agent(agent, df, n_episodes=5):
    """Évalue un agent sur n_episodes"""
    returns = []

    for episode in range(n_episodes):
        env = CryptoTradingEnv(df, initial_balance=10000)
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    mean_return = np.mean(returns)
    return mean_return

def format_time(seconds):
    """Formate le temps en heures:minutes:secondes"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def main():
    print("="*60)
    print("🚀 Entraînement Multi-Agent RL pour Crypto Trading")
    print("="*60)

    # Charger les données
    print("\n📥 Chargement des données...")
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ BTC/USDT chargées: {len(df)} candles")
    except FileNotFoundError:
        print("❌ Erreur: btc_usdt.csv non trouvé")
        return

    # Créer les queues pour récupérer les résultats
    results_queue = Queue()

    # Démarrer l'entraînement en parallèle
    print("\n🎯 Démarrage de l'entraînement en parallèle...")
    start_total = time.time()

    processes = [
        Process(target=train_dqn, args=(df, results_queue)),
        Process(target=train_ppo, args=(df, results_queue)),
        Process(target=train_a2c, args=(df, results_queue))
    ]

    for p in processes:
        p.start()

    # Attendre que tous les processes se terminent
    for p in processes:
        p.join()

    total_time = time.time() - start_total

    # Récupérer les résultats
    results = {}
    while not results_queue.empty():
        result = results_queue.get()
        results[result['name']] = result

    # Afficher les résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS DE L'ENTRAÎNEMENT")
    print("="*60)

    if len(results) > 0:
        # Performance
        print("\n🏆 Performance (Mean Return sur 5 episodes):")
        perf_str = " | ".join([
            f"{results[agent]['name']}: {results[agent]['return']:+.2f}€"
            for agent in sorted(results.keys())
        ])
        print(f"   {perf_str}")

        # Temps d'entraînement
        print("\n⏱️  Temps d'entraînement:")
        for agent_name in sorted(results.keys()):
            result = results[agent_name]
            if 'time' in result:
                time_str = format_time(result['time'])
                print(f"   {result['name']} training took {time_str}")

        print(f"\n   Total time: {format_time(total_time)}")

        # Meilleur agent
        best_agent = max(results.items(), key=lambda x: x[1].get('return', float('-inf')))
        print(f"\n🥇 Meilleur agent: {best_agent[0]} avec {best_agent[1]['return']:+.2f}€")

    print("\n" + "="*60)
    print("✨ Entraînement terminé!")
    print("="*60)

if __name__ == '__main__':
    main()
