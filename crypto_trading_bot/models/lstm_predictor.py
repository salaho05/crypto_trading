import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("❌ TensorFlow n'est pas installé")
    print("   Installez avec: pip install tensorflow")
    sys.exit(1)

def prepare_data(df, lookback=60):
    """Prépare les données en séquences"""
    data = df['close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Reshape pour LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, X_test, y_train, y_test, scaler

def main():
    print("="*70)
    print("🧠 LSTM Model pour Prédiction de Prix")
    print("="*70)
    
    # Charger les données
    print("\n📥 Chargement des données...")
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ BTC/USDT chargées: {len(df)} candles")
    except FileNotFoundError:
        print("❌ Erreur: btc_usdt.csv non trouvé")
        return
    
    # Préparer les données
    print("\n📊 Préparation des données (séquences de 60 candles)...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, lookback=60)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Créer le modèle LSTM
    print("\n🔨 Construction du modèle LSTM...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    print("✅ Modèle compilé")
    
    # Entraîner
    print("\n🤖 Entraînement (10 epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    print("✅ Entraînement terminé")
    
    # Évaluer
    print("\n📈 Évaluation sur test set...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse scaling pour les vraies valeurs
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    
    # Calculer l'accuracy (% de prédictions dans ±5%)
    accuracy = np.sum(np.abs(y_pred_rescaled - y_test_rescaled) < (y_test_rescaled * 0.05)) / len(y_test_rescaled) * 100
    
    print(f"   MAE:  ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   Accuracy (±5%): {accuracy:.1f}%")
    
    # Sauvegarder le modèle
    print("\n💾 Sauvegarde du modèle...")
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model.h5')
    print("✅ Modèle sauvegardé: models/lstm_model.h5")
    
    print("\n" + "="*70)
    print(f"✨ LSTM Model: MAE=${mae:.1f}, Accuracy={accuracy:.0f}%")
    print("="*70)

if __name__ == '__main__':
    main()
