import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    """Environnement Gymnasium pour le trading de cryptomonnaies"""

    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.current_step = 0
        self.done = False

        # Action space: 0 = SELL, 1 = HOLD, 2 = BUY
        self.action_space = spaces.Discrete(3)

        # Observation space: [close, sma_20, sma_50, rsi, macd, atr, balance]
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        # Historique pour le calcul de reward
        self.portfolio_values = []

    def reset(self, seed=None):
        """Réinitialise l'environnement"""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.current_step = 0
        self.done = False
        self.portfolio_values = []

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Exécute une action dans l'environnement"""
        current_price = self.df.loc[self.current_step, 'close']

        # Exécuter l'action
        if action == 2:  # BUY
            amount = self.balance * 0.1  # Achète 10% du balance
            if amount > 0:
                self.crypto_held += amount / current_price
                self.balance -= amount

        elif action == 0:  # SELL
            quantity = self.crypto_held * 0.5  # Vend 50% des crypto
            if quantity > 0:
                self.balance += quantity * current_price
                self.crypto_held -= quantity

        elif action == 1:  # HOLD
            pass

        # Avancer dans le temps
        self.current_step += 1

        # Vérifier si done
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # Calculer la valeur du portefeuille
        portfolio_value = self.balance + (self.crypto_held * current_price)
        self.portfolio_values.append(portfolio_value)

        # Calculer la reward
        reward = portfolio_value - self.initial_balance

        # Observation, reward, done, truncated, info
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': portfolio_value,
            'current_price': current_price
        }

        return observation, float(reward), self.done, False, info

    def _get_observation(self):
        """Retourne l'observation actuelle"""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        row = self.df.iloc[self.current_step]

        observation = np.array([
            row['close'],
            row['SMA_20'],
            row['SMA_50'],
            row['RSI'],
            row['MACD'],
            row['ATR'],
            self.balance
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        """Affiche l'état courant"""
        if self.current_step >= len(self.df):
            return

        row = self.df.iloc[self.current_step]
        current_price = row['close']
        portfolio_value = self.balance + (self.crypto_held * current_price)

        print(f"\n--- Step {self.current_step} ---")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Crypto Held: {self.crypto_held:.6f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"P&L: ${portfolio_value - self.initial_balance:.2f}")


# TEST SIMPLE
if __name__ == '__main__':
    print("=" * 60)
    print("🧪 Test Environnement Crypto Trading")
    print("=" * 60)

    # Charger les données
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ Données BTC/USDT chargées: {len(df)} candles")

        # Créer l'environnement
        env = CryptoTradingEnv(df, initial_balance=10000)
        print(f"✅ Environnement créé avec balance initiale: $10,000")

        # Reset
        observation, info = env.reset()
        print(f"✅ Environnement réinitialisé")

        # 10 steps aléatoires
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()  # Action aléatoire
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward

            action_name = ['SELL', 'HOLD', 'BUY'][action]
            print(f"   Step {step+1}: {action_name} | Reward: {reward:+.2f} | Portfolio: ${info['portfolio_value']:.2f}")

            if done:
                print(f"   Episode terminé!")
                break

        print("\n" + "=" * 60)
        print(f"✨ Env working! Portfolio: ${info['portfolio_value']:.2f}€")
        print(f"   Total Reward: {total_reward:.2f}")
        print("=" * 60)

    except FileNotFoundError:
        print("❌ Erreur: btc_usdt.csv non trouvé")
        print("   Exécute d'abord: python data/download_data.py")
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
