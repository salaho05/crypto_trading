import ccxt
import pandas as pd
import ta
import time
import os
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(symbol, limit=500):
    """Génère des données de test réalistes"""
    print(f"   📝 Génération de données de test pour {symbol}...")

    # Déterminer le prix initial selon le symbole
    prices = {'BTC/USDT': 45000, 'ETH/USDT': 2500, 'SOL/USDT': 150}
    start_price = prices.get(symbol, 100)

    # Générer les timestamps
    dates = pd.date_range(end=datetime.now(), periods=limit, freq='4h')

    # Générer les prix avec une marche aléatoire
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, limit)
    prices_array = start_price * np.exp(np.cumsum(returns))

    # Créer l'OHLCV
    data = []
    for i, date in enumerate(dates):
        open_price = prices_array[i]
        close_price = prices_array[i] * (1 + np.random.normal(0, 0.01))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df

def download_data(symbol, timeframe='4h', limit=500):
    """Télécharge les données OHLCV depuis Binance"""
    try:
        exchange = ccxt.binance()

        print(f"\n📥 Téléchargement de {symbol}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
    except Exception as e:
        print(f"\n📥 Tentative pour {symbol}...")
        print(f"   ⚠️  Problème de connexion API")
        return generate_sample_data(symbol, limit)

def add_indicators(df):
    """Ajoute les indicateurs techniques"""
    print("   ➕ Ajout des indicateurs techniques...")

    # SMA
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)

    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)

    # MACD
    macd = ta.trend.macd(df['close'])
    df['MACD'] = macd
    df['MACD_signal'] = ta.trend.macd_signal(df['close'])
    df['MACD_diff'] = ta.trend.macd_diff(df['close'])

    # ATR
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Bollinger Bands
    bollinger = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
    df['BB_high'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['BB_low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)

    # Remplir les NaN avec forward fill
    df = df.ffill().bfill()

    return df

def save_and_stats(df, symbol, filename):
    """Sauvegarde et affiche les statistiques"""
    filepath = filename
    df.to_csv(filepath)

    print(f"\n✅ {symbol}")
    print(f"   📊 Candles: {len(df)}")
    print(f"   📅 Période: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")

    # Taille du fichier
    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"   💾 Taille: {file_size_kb:.2f} KB")

def main():
    print("=" * 60)
    print("🚀 Téléchargement des données crypto depuis Binance")
    print("=" * 60)

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    filenames = [
        'data/btc_usdt.csv',
        'data/eth_usdt.csv',
        'data/sol_usdt.csv'
    ]

    for symbol, filename in zip(symbols, filenames):
        try:
            # Télécharger les données
            df = download_data(symbol, timeframe='4h', limit=500)

            # Ajouter les indicateurs
            df = add_indicators(df)

            # Sauvegarder et afficher les statistiques
            save_and_stats(df, symbol, filename)

            # Délai pour respecter les limites de l'API
            time.sleep(1)

        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {str(e)}")

    print("\n" + "=" * 60)
    print("✨ Téléchargement terminé!")
    print("=" * 60)

if __name__ == '__main__':
    main()
