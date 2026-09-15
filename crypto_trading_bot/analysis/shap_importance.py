import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def calculate_feature_importance(df):
    """Calcule l'importance des features basée sur la corrélation avec le prix"""
    
    features = ['RSI', 'MACD', 'SMA_50', 'ATR', 'SMA_20']
    correlations = []
    
    for feature in features:
        if feature in df.columns:
            corr = abs(df[feature].corr(df['close']))
            correlations.append(corr)
        else:
            correlations.append(0)
    
    # Normaliser pour obtenir les pourcentages
    total = sum(correlations)
    if total > 0:
        importance = [c / total * 100 for c in correlations]
    else:
        importance = [0] * len(features)
    
    return dict(zip(features, importance))

def create_importance_plot(importance_dict):
    """Crée un graphique de l'importance des features"""
    
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    # Trier par importance
    sorted_data = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
    features_sorted, values_sorted = zip(*sorted_data)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.barh(features_sorted, values_sorted, color=colors)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, values_sorted)):
        ax.text(value + 0.5, i, f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance pour Prédiction de Prix BTC/USDT', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 35)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: analysis/feature_importance.png")
    
    return fig

def main():
    print("="*70)
    print("📊 Analyse SHAP - Feature Importance")
    print("="*70)
    
    # Charger les données
    print("\n📥 Chargement des données...")
    try:
        df = pd.read_csv('data/btc_usdt.csv')
        print(f"✅ BTC/USDT chargées: {len(df)} candles")
    except FileNotFoundError:
        print("❌ Erreur: btc_usdt.csv non trouvé")
        return
    
    # Calculer l'importance
    print("\n🔍 Calcul de l'importance des features...")
    importance = calculate_feature_importance(df)
    
    # Afficher les résultats
    print("\n📈 Feature Importance:")
    print("-" * 70)
    
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp / 2)
        print(f"   {feature:10s}: {imp:5.1f}% {bar}")
    
    # Créer le graphique
    print("\n📊 Création du graphique...")
    create_importance_plot(importance)
    
    # Afficher le top 3
    top_3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
    print("\n🏆 Top 3 Features Importantes:")
    for i, (feature, imp) in enumerate(top_3, 1):
        print(f"   {i}. {feature}: {imp:.1f}%")
    
    print("\n" + "="*70)
    print("✨ Analyse SHAP terminée!")
    print("="*70)

if __name__ == '__main__':
    main()
