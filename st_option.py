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

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# ----------- Données de base -----------
def get_last_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Erreur get_last_price: {e}")
        return None

def get_name(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("shortName", "Nom inconnu")
    except Exception as e:
        print(f"Erreur get_name: {e}")
        return "Erreur"

def variation_depuis_ouverture(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        open_price = data['Open'].iloc[-1]
        close_price = data['Close'].iloc[-1]
        return round((close_price - open_price) / open_price * 100, 2)
    except:
        return None

def volume_depuis_ouverture(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return int(data['Volume'].iloc[-1])
    except:
        return None

def volume_moyen(ticker, period="1mo"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        return int(data['Volume'].mean())
    except:
        return None

# ----------- Options -----------
def get_options_dataframe(ticker):
    try:
        stock = yf.Ticker(ticker)
        dates = stock.options
        if not dates:
            return pd.DataFrame(), "Aucune échéance trouvée."

        opt_chain = stock.option_chain(dates[0])
        calls = opt_chain.calls.copy()
        calls["T"] = (pd.to_datetime(dates[0]) - pd.Timestamp.today()).days / 365
        return calls, ""
    except Exception as e:
        return pd.DataFrame(), f"Erreur get_options_dataframe: {e}"

# ----------- Black-Scholes -----------
def black_scholes(ticker, K, T, r, sigma, option_type):
    try:
        if sigma <= 0 or T <= 0:
            return None
        S = get_last_price(ticker)
        if S is None:
            return None

        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            return None
    except Exception as e:
        print(f"Erreur black_scholes: {e}")
        return None

def get_implied_volatility(ticker, K, T, r, market_price, option_type):
    def objective(sigma):
        price = black_scholes(ticker, K, T, r, sigma, option_type)
        return (price - market_price) if price is not None else np.inf

    try:
        result = root_scalar(objective, bracket=[1e-5, 5], method='brentq')
        return result.root if result.converged else None
    except Exception as e:
        print(f"Erreur get_implied_volatility: {e}")
        return None

# ----------- Visualisation -----------
def plot_vol_surface(df, ticker, r, option_type="call"):
    try:
        def iv_row(row):
            iv = get_implied_volatility(ticker, row['strike'], row['T'], r, row['lastPrice'], option_type)
            return iv if iv is not None else np.nan

        df['impliedVol'] = df.apply(iv_row, axis=1)
        df.dropna(subset=['impliedVol'], inplace=True)

        X = df['strike']
        Y = df['T']
        Z = df['impliedVol']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='viridis')
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturité")
        ax.set_zlabel("Volatilité implicite")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur dans le tracé de la surface : {e}")


def plot_smile_and_skew(df, ticker, r, option_type="call"):
    try:
        df['impliedVol'] = df.apply(
            lambda row: get_implied_volatility(ticker, row['strike'], row['T'], r, row['lastPrice'], option_type),
            axis=1)
        df.dropna(subset=['impliedVol'], inplace=True)

        fig, ax = plt.subplots()
        ax.plot(df['strike'], df['impliedVol'], 'bo-', label="Smile")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Volatilité implicite")
        ax.set_title("Volatility Smile")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur dans le tracé du smile/skew : {e}")