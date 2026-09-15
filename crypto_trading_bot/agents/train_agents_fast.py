import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from env.crypto_trading_env import CryptoTradingEnv
import time
import warnings

warnings.filterwarnings('ignore')

os.makedirs('agents/models', exist_ok=True)

INITIAL_BALANCE = 10000

def evaluate_agent(agent, df, n_episodes=3):
    """Évalue un agent"""
    returns = []
    for episode in range(n_episodes):
        env = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        returns.append((info['portfolio_value'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100)
    return np.mean(returns)

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"

def main():
    print("="*70)
    print("🚀 Multi-Agent RL Training (Fast Version)")
    print("="*70)
    
    # Charger données
    print("\n📥 Chargement BTC/USDT...")
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ {len(df)} candles")
    except FileNotFoundError:
        print("❌ btc_usdt.csv non trouvé")
        return
    
    results = {}
    total_start = time.time()
    
    # DQN
    print("\n" + "="*70)
    print("🤖 Training DQN (20000 timesteps)...")
    print("="*70)
    start = time.time()
    
    env = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
    env = DummyVecEnv([lambda: env])
    dqn = DQN('MlpPolicy', env, learning_rate=0.0001, verbose=0, device='cpu')
    dqn.learn(total_timesteps=20000)
    dqn.save('agents/models/dqn_model')
    
    dqn_return = evaluate_agent(dqn, df)
    dqn_time = time.time() - start
    results['DQN'] = {'return': dqn_return, 'time': dqn_time}
    print(f"✅ DQN Return: {dqn_return:+.2f}% | Time: {format_time(dqn_time)}")
    
    # PPO
    print("\n" + "="*70)
    print("🤖 Training PPO (20000 timesteps)...")
    print("="*70)
    start = time.time()
    
    env = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
    env = DummyVecEnv([lambda: env])
    ppo = PPO('MlpPolicy', env, learning_rate=0.0003, verbose=0, device='cpu')
    ppo.learn(total_timesteps=20000)
    ppo.save('agents/models/ppo_model')
    
    ppo_return = evaluate_agent(ppo, df)
    ppo_time = time.time() - start
    results['PPO'] = {'return': ppo_return, 'time': ppo_time}
    print(f"✅ PPO Return: {ppo_return:+.2f}% | Time: {format_time(ppo_time)}")
    
    # A2C
    print("\n" + "="*70)
    print("🤖 Training A2C (20000 timesteps)...")
    print("="*70)
    start = time.time()
    
    env = CryptoTradingEnv(df, initial_balance=INITIAL_BALANCE)
    env = DummyVecEnv([lambda: env])
    a2c = A2C('MlpPolicy', env, learning_rate=0.0007, verbose=0, device='cpu')
    a2c.learn(total_timesteps=20000)
    a2c.save('agents/models/a2c_model')
    
    a2c_return = evaluate_agent(a2c, df)
    a2c_time = time.time() - start
    results['A2C'] = {'return': a2c_return, 'time': a2c_time}
    print(f"✅ A2C Return: {a2c_return:+.2f}% | Time: {format_time(a2c_time)}")
    
    total_time = time.time() - total_start
    
    # Résultats
    print("\n" + "="*70)
    print("📊 RÉSULTATS")
    print("="*70)
    
    print("\n🏆 Performance:")
    for agent, data in results.items():
        print(f"   {agent}: {data['return']:+.2f}%")
    
    print("\n⏱️  Temps d'entraînement:")
    for agent, data in results.items():
        print(f"   {agent}: {format_time(data['time'])}")
    
    best = max(results.items(), key=lambda x: x[1]['return'])
    print(f"\n🥇 Meilleur: {best[0]} avec {best[1]['return']:+.2f}%")
    print(f"\n   Total Time: {format_time(total_time)}")
    
    print("\n" + "="*70)
    print("✨ Training terminé!")
    print("="*70)

if __name__ == '__main__':
    main()
