import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.crypto_trading_env import CryptoTradingEnv
import warnings

warnings.filterwarnings('ignore')

class WalkForwardValidator:
    """Validateur Walk-Forward pour stratégies de trading"""

    def __init__(self, df, n_windows=3, train_pct=0.70, test_pct=0.20):
        self.df = df.reset_index(drop=True)
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.test_pct = test_pct

        # Calculer la taille des fenêtres
        total_len = len(df)
        window_size = int(total_len / (n_windows + 1))
        self.train_size = int(window_size * train_pct)
        self.test_size = int(window_size * test_pct)

        self.results = []

    def calculate_metrics(self, env, agent, df_test, initial_balance=10000):
        """Calcule les métriques de performance"""

        # Exécuter l'agent sur les données de test
        obs, _ = env.reset()
        trades = []
        portfolio_values = [initial_balance]

        done = False
        step = 0
        while not done and step < len(df_test) - 1:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info['portfolio_value'])

            # Enregistrer les trades
            if action == 2:  # BUY
                trades.append({'type': 'BUY', 'price': info['current_price']})
            elif action == 0:  # SELL
                trades.append({'type': 'SELL', 'price': info['current_price']})

            step += 1

        # Calculer les métriques
        portfolio_values = np.array(portfolio_values)
        final_value = portfolio_values[-1]

        # Total Return
        total_return = (final_value - initial_balance) / initial_balance * 100

        # Sharpe Ratio
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        annual_return = np.mean(daily_returns) * 365 * 100
        annual_volatility = np.std(daily_returns) * np.sqrt(365) * 100
        sharpe_ratio = (annual_return - 2) / (annual_volatility + 1e-6)

        # Max Drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)

        # Win Rate
        profitable_trades = 0
        total_trades = 0

        for i in range(1, len(trades)):
            if trades[i-1]['type'] == 'BUY' and trades[i]['type'] == 'SELL':
                if trades[i]['price'] > trades[i-1]['price']:
                    profitable_trades += 1
                total_trades += 1

        win_rate = (profitable_trades / max(total_trades, 1)) * 100

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': final_value,
            'trades': len(trades)
        }

    def calculate_buy_hold_return(self, df_test):
        """Calcule le retour du Buy-and-Hold baseline"""
        initial_price = df_test['close'].iloc[0]
        final_price = df_test['close'].iloc[-1]
        return (final_price - initial_price) / initial_price * 100

    def validate(self):
        """Lance la validation walk-forward"""
        print("="*70)
        print("📊 Walk-Forward Validation")
        print("="*70)

        window_size = int(len(self.df) / (self.n_windows + 1))
        initial_balance = 10000

        for window_idx in range(self.n_windows):
            print(f"\n🔄 Window {window_idx + 1}/{self.n_windows}")
            print("-" * 70)

            # Créer les splits
            start_idx = window_idx * window_size
            train_end_idx = start_idx + self.train_size
            test_end_idx = train_end_idx + self.test_size

            df_train = self.df.iloc[start_idx:train_end_idx].reset_index(drop=True)
            df_test = self.df.iloc[train_end_idx:test_end_idx].reset_index(drop=True)

            print(f"   Train: {len(df_train)} candles ({start_idx} → {train_end_idx})")
            print(f"   Test:  {len(df_test)} candles ({train_end_idx} → {test_end_idx})")

            # Vérifier s'il y a assez de données
            if len(df_train) < 50 or len(df_test) < 20:
                print(f"   ⚠️  Données insuffisantes, skipping...")
                continue

            # Entraîner PPO
            print(f"   🤖 Training PPO...")
            env = CryptoTradingEnv(df_train, initial_balance=initial_balance)
            env = DummyVecEnv([lambda: env])

            agent = PPO(
                'MlpPolicy',
                env,
                learning_rate=0.0003,
                verbose=0,
                device='cpu'
            )

            agent.learn(total_timesteps=10000)

            # Tester sur données neuves
            print(f"   📈 Testing on unseen data...")
            env_test = CryptoTradingEnv(df_test, initial_balance=initial_balance)

            metrics = self.calculate_metrics(env_test, agent, df_test, initial_balance)

            # Buy-and-Hold baseline
            bh_return = self.calculate_buy_hold_return(df_test)

            # Afficher résultats
            print(f"\n   📊 Résultats Window {window_idx + 1}:")
            print(f"      Total Return:      {metrics['total_return']:+.2f}%")
            print(f"      Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
            print(f"      Max Drawdown:      {metrics['max_drawdown']:.2f}%")
            print(f"      Win Rate:          {metrics['win_rate']:.1f}%")
            print(f"      Trades:            {metrics['trades']}")
            print(f"      Final Portfolio:   ${metrics['final_value']:,.2f}")
            print(f"\n   📈 Buy-and-Hold Return: {bh_return:+.2f}%")
            print(f"   {'✅ Agent outperforms' if metrics['total_return'] > bh_return else '❌ Agent underperforms'}")

            metrics['window'] = window_idx + 1
            metrics['bh_return'] = bh_return
            self.results.append(metrics)

    def print_summary(self):
        """Affiche le résumé des résultats"""
        if len(self.results) == 0:
            print("\n❌ Pas de résultats à afficher")
            return

        print("\n" + "="*70)
        print("📊 RÉSUMÉ DES RÉSULTATS")
        print("="*70)

        # Moyenne des résultats
        avg_return = np.mean([r['total_return'] for r in self.results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in self.results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in self.results])
        avg_win_rate = np.mean([r['win_rate'] for r in self.results])
        avg_bh = np.mean([r['bh_return'] for r in self.results])

        print(f"\n🤖 Agent Performance (Average):")
        print(f"   Total Return:       {avg_return:+.2f}%")
        print(f"   Sharpe Ratio:       {avg_sharpe:.2f}")
        print(f"   Max Drawdown:       {avg_drawdown:.2f}%")
        print(f"   Win Rate:           {avg_win_rate:.1f}%")

        print(f"\n📈 Buy-and-Hold Baseline (Average):")
        print(f"   Return:             {avg_bh:+.2f}%")

        print(f"\n🏆 Comparison:")
        outperformance = avg_return - avg_bh
        if outperformance > 0:
            print(f"   ✅ Agent outperforms by {outperformance:+.2f}%")
        else:
            print(f"   ❌ Agent underperforms by {outperformance:.2f}%")

        print(f"\n📋 Per-Window Results:")
        for result in self.results:
            window = result['window']
            agent_ret = result['total_return']
            bh_ret = result['bh_return']
            print(f"   Window {window}: Agent {agent_ret:+.2f}% vs Buy-Hold {bh_ret:+.2f}%")

        print("\n" + "="*70)
        print("✨ Validation terminée!")
        print("="*70)


def main():
    print("="*70)
    print("🚀 Walk-Forward Validation pour Crypto Trading")
    print("="*70)

    # Charger les données
    print("\n📥 Chargement des données...")
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ BTC/USDT chargées: {len(df)} candles")
    except FileNotFoundError:
        print("❌ Erreur: btc_usdt.csv non trouvé")
        return

    # Lancer la validation
    validator = WalkForwardValidator(df, n_windows=3, train_pct=0.70, test_pct=0.20)
    validator.validate()
    validator.print_summary()


if __name__ == '__main__':
    main()
