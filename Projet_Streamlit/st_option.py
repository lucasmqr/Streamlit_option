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
import time 
from datetime import datetime
# ----------- Données de base -----------


def convert_to_paris_string(ts):

    # Convertir vers heure de Paris
    ts_paris = ts.tz_convert('Europe/Paris')

    # Formater en jj/mm/aaaa : hh:mm
    return ts_paris.strftime("%d/%m/%Y : %H:%M")

def get_last_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        print(data)
        return data['Close'].iloc[-1],convert_to_paris_string(data.index[-1])
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

def get_market_cap(ticker):
    time.sleep(0.5)
    stock=yf.Ticker(ticker)
    market_cap=stock.info.get("marketCap")
    return market_cap

def get_sector(ticker):

    stock=yf.Ticker(ticker)
    sector = stock.info.get("sector")

    return sector

def get_website(ticker):
    stock=yf.Ticker(ticker)
    return (stock.info.get("website"))

def get_description(ticker):
    stock=yf.Ticker(ticker)
    return(stock.info.get("longBusinessSummary"))

def get_beta(ticker):
    stock=yf.Ticker(ticker)
    return(stock.info.get("beta"))

def get_fwpe(ticker):
    stock=yf.Ticker(ticker)
    return(stock.info.get("forwardPE"))


def variation_depuis_ouverture(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        open_price = data['Open'].iloc[-1]
        close_price = data['Close'].iloc[-1]
        return round((close_price - open_price) / open_price * 100, 2)
    except:
        return None
    
def performance_ytd(ticker):
    try:
        today = datetime.now().date()
        start_of_year = datetime(today.year, 1, 1).date()

        data = yf.Ticker(ticker).history(start=start_of_year, end=today)
        if data.empty or 'Close' not in data:
            return None

        start_price = data['Close'].iloc[0]
        last_price = data['Close'].iloc[-1]

        return round((last_price - start_price) / start_price * 100, 2)
    except Exception as e:
        print(f"Erreur : {e}")
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
    
def get_stock_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Récupère l'historique du prix de clôture d'une action sur une période donnée.
    
    :param ticker: Le ticker de l'action (ex: "AAPL")
    :param period: La période (ex: "6mo", "1y", "ytd", "max")
    :return: Un DataFrame contenant les prix de clôture avec l'index en datetime
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return pd.DataFrame()
        return data[["Close"]]
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return pd.DataFrame()

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
def black_scholes(S,ticker, K, T, r, sigma, option_type):
    try:
        if sigma <= 0 or T <= 0:
            return None
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

def get_implied_volatility(S, K, T, r, market_price, option_type):
    def objective(sigma):
        price = black_scholes(S, None, K, T, r, sigma, option_type)
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






def get_all_options(ticker: str) -> pd.DataFrame:
    """
    Récupère toutes les options (calls et puts) disponibles pour toutes les dates d'échéance.
    
    :param ticker: Le ticker de l'action (ex: "AAPL")
    :return: Un DataFrame avec les données des options, contenant une colonne 'expiration' et 'optionType'
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options

        all_options = []
        for date in expirations:
            opt_chain = stock.option_chain(date)
            
            calls = opt_chain.calls.copy()
            calls["optionType"] = "call"
            calls["expiration"] = date

            puts = opt_chain.puts.copy()
            puts["optionType"] = "put"
            puts["expiration"] = date

            all_options.append(calls)
            all_options.append(puts)

        df_all = pd.concat(all_options, ignore_index=True)
        return df_all

    except Exception as e:
        print(f"Erreur lors de la récupération des options pour {ticker} : {e}")
        return pd.DataFrame()
    
def black_scholes_greeks(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0 or S <= 0:
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    greeks = {}

    if option_type == "call":
        greeks['Delta'] = norm.cdf(d1)
        greeks['Theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                           r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        greeks['Rho'] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == "put":
        greeks['Delta'] = norm.cdf(d1) - 1
        greeks['Theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                           r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        greeks['Rho'] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        return None

    greeks['Vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100  # en pourcentage

    return greeks
    


# tickers_dict = {
#     "USA": {
#         "3M": "^IRX",
#         "5Y": "^FVX",
#         "10Y": "^TNX",
#         "30Y": "^TYX"
#     },
#     "France": {
#         "2Y": "FR2YT=RR",
#         "5Y": "FR5YT=RR",
#         "10Y": "FR10YT=RR",
#         "30Y": "FR30YT=RR"
#     },
#     "Germany": {
#         "2Y": "DE2YT=RR",
#         "5Y": "DE5YT=RR",
#         "10Y": "DE10YT=RR",
#         "30Y": "DE30YT=RR"
#     }
# }


# def get_yield_curve(tickers_dict, start_date, end_date):
#     data = {}
#     for country, tickers in tickers_dict.items():
#         curves = {}
#         for label, ticker in tickers.items():
#             hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
#             if not hist.empty:
#                 curves[label] = {
#                     'Début année': hist['Close'].iloc[0],
#                     'Dernier': hist['Close'].iloc[-1]
#                 }
#         df = pd.DataFrame(curves).T  # maturités en index, colonnes = Début année, Dernier
#         df.index.name = "Maturité"
#         data[country] = df
#     return data


# def plot_yield_curves(data):
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for country, df in data.items():
#         print(f"Colonnes pour {country}:", df.columns)  # <--- debug
#         ax.plot(df.index, df['Début année'], '--', label=f"{country} - Début année")
#         ax.plot(df.index, df['Dernier'], '-', label=f"{country} - Dernier")

#     ax.set_xlabel("Maturité")
#     ax.set_ylabel("Taux (%)")
#     ax.set_title("Courbes de taux - Début d'année vs Dernier")
#     ax.legend()
#     st.pyplot(fig)
