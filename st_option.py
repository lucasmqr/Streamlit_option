"""
Création d'un projet Streamlit

fonction à faire :


Fonction qui retourne :

Partie générale (présentation)

- le Nom de l’entreprise (+ afficher le Ticker)
- Prix actuel	
- Le dernier prix de l’action
- Variation (%)	
- Hausse ou baisse depuis la dernière clôture
- Volume	Nombre d’actions échangées pendant la journée
- volume moyen (pour détecter les anomalies) 
- Faire une moyenne mobile 
- Market Cap	Capitalisation boursière (valeur totale de l’entreprise)


Partie Option

- Récupérer la volatilité implicite avec le modèle de Black Scholes pour une option donnée 
- Représenter la nappe de vol 
- Représenter le smile et le skew
- Calcul des grecques + réprésentation 
- Faire un système où on combine des optines et ca trace le payoff 

Partie Mémoire : 

- Heston où on peut simuler le prix d'option 


Parie Bond 

- Faire une courbe des taux 
- Comparer les taux des principaux pays à leur moyenne mobile 200j et 50j
- Faire une courbe des CDS
- Récupérer le risque de défaut implicite comme dans le cours de Lorenz


"""

# Import des bibliothèques utiles
import yfinance as yf
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.dates as mdates
from scipy.interpolate import griddata
import seaborn as sns
import time  # ✅ pour gérer les délais entre les requêtes

# ---------------- PARTIE GÉNÉRALE ----------------

def get_last_price(ticker):
    time.sleep(1)
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker '{ticker}'")
        return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Erreur lors de la récupération du prix pour {ticker}: {e}")
        return None

def get_name(ticker):
    time.sleep(1)
    stock = yf.Ticker(ticker)
    return stock.info.get('longName')

def variation_depuis_ouverture(ticker):
    time.sleep(1)
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="5m")
    if data.empty:
        return "Données non disponibles"
    ouverture = data['Open'][0]
    dernier_cours = data['Close'][-1]
    variation = ((dernier_cours - ouverture) / ouverture) * 100
    return round(variation, 2)

def volume_moyen(ticker, periode):
    time.sleep(1)
    stock = yf.Ticker(ticker)
    data = stock.history(period=periode)
    if data.empty:
        return "Données non disponibles"
    return int(data['Volume'].mean())

def volume_depuis_ouverture(ticker):
    time.sleep(1)
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="5m")
    if data.empty:
        return "Données non disponibles"
    return int(data['Volume'].sum())

def get_market_cap(ticker):
    time.sleep(1)
    stock = yf.Ticker(ticker)
    return stock.info.get("marketCap", "Non disponible")

# ---------------- PARTIE OPTIONS ----------------

def black_scholes(ticker, K, T, r, sigma, option_type='call'):
    time.sleep(0.5)
    S = get_last_price(ticker)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return "Paramètres invalides"
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        prix = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        prix = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return "Type d'option non reconnu ('call' ou 'put')"
    return round(prix, 4)

def get_options_dataframe(ticker):
    time.sleep(1)
    stock = yf.Ticker(ticker)
    try:
        expiration_dates = stock.options
        if not expiration_dates:
            return pd.DataFrame(), "Aucune option disponible pour ce ticker."
    except Exception as e:
        return pd.DataFrame(), f"Erreur lors de la récupération des options : {e}"

    all_options = []
    for date in expiration_dates:
        time.sleep(0.5)
        try:
            calls = stock.option_chain(date).calls
            puts = stock.option_chain(date).puts
            calls['type'] = 'call'
            puts['type'] = 'put'
            combined = pd.concat([calls, puts])
            combined['expirationDate'] = date
            all_options.append(combined)
        except Exception as e:
            print(f"Erreur pour la date {date} : {e}")
            continue

    if not all_options:
        return pd.DataFrame(), "Aucune donnée d'options trouvée."
    return pd.concat(all_options, ignore_index=True), "Succès"

def get_implied_volatility(ticker, K, T, r, market_price, option_type='call', sigma_init=0.2, tol=1e-6, max_iter=100):
    time.sleep(0.5)
    sigma = sigma_init
    S = get_last_price(ticker)
    for i in range(max_iter):
        price = black_scholes(ticker, K, T, r, sigma, option_type)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega == 0:
            break
        diff = price - market_price
        sigma -= diff / vega
        if abs(diff) < tol:
            return round(sigma, 6)
    return None

def plot_vol_surface(df_options, ticker, r, option_type='call'):
    time.sleep(0.5)
    df = df_options[df_options['type'] == option_type].copy()
    if df.empty:
        print("Aucune option de type", option_type)
        return
    if not np.issubdtype(df['expirationDate'].dtype, np.datetime64):
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
    today = pd.Timestamp.today()
    df['T'] = (df['expirationDate'] - today).dt.total_seconds() / (365.25*24*3600)
    df = df[df['T'] > 0]
    stock = yf.Ticker(ticker)
    S = stock.history(period="1d")['Close'][-1]
    def iv_row(row):
        return get_implied_volatility(ticker, row['strike'], row['T'], r, row['lastPrice'], option_type)
    df['impliedVol'] = df.apply(iv_row, axis=1)
    df = df.dropna(subset=['impliedVol'])
    X, Y, Z = df['strike'].values, df['T'].values, df['impliedVol'].values
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((X, Y), Z, (XI, YI), method='linear')
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XI, YI, ZI, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturité (années)')
    ax.set_zlabel('Volatilité implicite')
    ax.set_title(f"Nappe de volatilité implicite pour {ticker} ({option_type})")
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Volatilité implicite')
    plt.show()

def plot_smile_and_skew(df_options, ticker, r, option_type='call', spot=None):
    time.sleep(0.5)
    sns.set(style="whitegrid")
    df = df_options[df_options['type'] == option_type].copy()
    if df.empty:
        print("Aucune option du type demandé")
        return
    df['expirationDate'] = pd.to_datetime(df['expirationDate'])
    today = pd.Timestamp.today()
    df['T'] = (df['expirationDate'] - today).dt.total_seconds() / (365.25 * 24 * 3600)
    df = df[df['T'] > 0]
    if spot is None:
        spot = get_last_price(ticker)
        if spot is None:
            print("Impossible d'obtenir le prix du sous-jacent.")
            return
    def iv_row(row):
        return get_implied_volatility(ticker, row['strike'], row['T'], r, row['lastPrice'], option_type)
    df['impliedVol'] = df.apply(iv_row, axis=1)
    df = df.dropna(subset=['impliedVol'])
    unique_T = sorted(df['T'].unique())
    median_T = unique_T[len(unique_T)//2]
    smile_df = df[np.isclose(df['T'], median_T, atol=0.01)]
    closest_strike = df.loc[(df['strike'] - spot).abs().idxmin(), 'strike']
    skew_df = df[np.isclose(df['strike'], closest_strike, atol=1)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(smile_df['strike'], smile_df['impliedVol'], marker='o')
    axs[0].set_title(f"Volatility Smile (T ≈ {round(median_T, 3)} ans)")
    axs[0].set_xlabel("Strike")
    axs[0].set_ylabel("Volatilité implicite")
    axs[0].axvline(spot, color='gray', linestyle='--', label='Spot')
    axs[0].legend()
    axs[1].plot(skew_df['T'], skew_df['impliedVol'], marker='o', color='orange')
    axs[1].set_title(f"Volatility Skew (Strike ≈ {closest_strike})")
    axs[1].set_xlabel("Maturité (années)")
    axs[1].set_ylabel("Volatilité implicite")
    plt.suptitle(f"Smile & Skew pour {ticker} ({option_type})", fontsize=14)
    plt.tight_layout()
    plt.show()
