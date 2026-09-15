import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from env.crypto_trading_env import CryptoTradingEnv
import warnings

warnings.filterwarnings('ignore')

# Variables globales
df = None
INITIAL_BALANCE = 10000

def evaluate_agent(agent, df_eval, n_episodes=5):
    """Évalue un agent sur n_episodes"""
    returns = []
    
    for episode in range(n_episodes):
        env = CryptoTradingEnv(df_eval, initial_balance=INITIAL_BALANCE)
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        final_return = (info['portfolio_value'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        returns.append(final_return)
    
    return np.mean(returns)

def objective(trial):
    """Fonction objectif pour Optuna"""
    
    # Proposer les hyperparamètres
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    
    print(f"\n   Trial {trial.number}:")
    print(f"      lr={learning_rate:.6f}, gamma={gamma:.4f}, batch_size={batch_size}, clip={clip_range:.2f}")
    
    try:
        # Créer l'environnement
        env = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
        env = DummyVecEnv([lambda: env])
        
        # Entraîner PPO avec ces hyperparamètres
        agent = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            clip_range=clip_range,
            verbose=0,
            device='cpu',
            n_steps=2048
        )
        
        agent.learn(total_timesteps=15000)
        
        # Évaluer
        mean_return = evaluate_agent(agent, df, n_episodes=3)
        
        print(f"      → Return: {mean_return:+.2f}%")
        
        return mean_return
        
    except Exception as e:
        print(f"      ❌ Erreur: {str(e)}")
        return float('-inf')

def main():
    global df
    
    print("="*70)
    print("🔧 Optimisation des Hyperparamètres avec Optuna")
    print("="*70)
    
    # Charger les données
    print("\n📥 Chargement des données...")
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ BTC/USDT chargées: {len(df)} candles")
    except FileNotFoundError:
        print("❌ Erreur: btc_usdt.csv non trouvé")
        return
    
    # Évaluer un agent original (baseline)
    print("\n📊 Évaluation du PPO Original (hyperparams par défaut)...")
    print("-" * 70)
    
    env_baseline = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
    env_baseline = DummyVecEnv([lambda: env_baseline])
    
    agent_baseline = PPO(
        'MlpPolicy',
        env_baseline,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=64,
        clip_range=0.2,
        verbose=0,
        device='cpu'
    )
    
    agent_baseline.learn(total_timesteps=15000)
    baseline_return = evaluate_agent(agent_baseline, df, n_episodes=5)
    print(f"✅ Original PPO Return: {baseline_return:+.2f}%")
    
    # Créer et optimiser l'étude Optuna
    print("\n" + "="*70)
    print("🔍 Optimisation avec Optuna (20 trials)...")
    print("="*70)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE L'OPTIMISATION")
    print("="*70)
    
    best_trial = study.best_trial
    
    print(f"\n🏆 Best Trial: Trial {best_trial.number}")
    print(f"\n   Hyperparamètres optimaux:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.6f}" if value < 0.01 else f"      {key}: {value:.4f}")
        else:
            print(f"      {key}: {value}")
    
    print(f"\n   Best Value: {best_trial.value:+.2f}%")
    
    # Entraîner un nouvel agent avec les meilleurs hyperparams
    print("\n" + "="*70)
    print("🤖 Entraînement du PPO Optimisé...")
    print("="*70)
    
    env_optimized = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
    env_optimized = DummyVecEnv([lambda: env_optimized])
    
    agent_optimized = PPO(
        'MlpPolicy',
        env_optimized,
        learning_rate=best_trial.params['learning_rate'],
        gamma=best_trial.params['gamma'],
        batch_size=best_trial.params['batch_size'],
        clip_range=best_trial.params['clip_range'],
        verbose=0,
        device='cpu',
        n_steps=2048
    )
    
    agent_optimized.learn(total_timesteps=20000)
    optimized_return = evaluate_agent(agent_optimized, df, n_episodes=5)
    
    print(f"✅ Optimized PPO Return: {optimized_return:+.2f}%")
    
    # Comparaison
    print("\n" + "="*70)
    print("📈 COMPARAISON")
    print("="*70)
    
    improvement = optimized_return - baseline_return
    improvement_pct = (improvement / abs(baseline_return)) * 100 if baseline_return != 0 else 0
    
    print(f"\n   Original PPO:   {baseline_return:+.2f}%")
    print(f"   Optimized PPO:  {optimized_return:+.2f}%")
    print(f"   Improvement:    {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"   Gain:           +{improvement_pct:.1f}% 🎉")
    else:
        print(f"   Loss:           {improvement_pct:.1f}%")
    
    # Statistiques de l'étude
    print("\n" + "="*70)
    print("📊 STATISTIQUES DE L'ÉTUDE")
    print("="*70)
    
    print(f"\n   Total Trials:   {len(study.trials)}")
    print(f"   Best Trial:     Trial {best_trial.number}")
    print(f"   Best Value:     {best_trial.value:+.2f}%")
    
    # Top 5 trials
    print(f"\n   Top 5 Trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    for i, trial in enumerate(trials_sorted[:5], 1):
        print(f"      {i}. Trial {trial.number}: {trial.value:+.2f}%")
    
    print("\n" + "="*70)
    print("✨ Optimisation terminée!")
    print("="*70)
    
    try:
        os.makedirs('optimization/models', exist_ok=True)
        agent_optimized.save('optimization/models/ppo_optimized')
        print("\n💾 Modèle optimisé sauvegardé: optimization/models/ppo_optimized")
    except Exception as e:
        print(f"\n⚠️  Erreur lors de la sauvegarde: {str(e)}")


if __name__ == '__main__':
    main()
