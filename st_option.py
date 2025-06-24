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

def last_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker '{ticker}'")
        return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Erreur lors de la récupération du prix pour {ticker}: {e}")
        return None
    

def volume_total(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError("Données introuvables.")
        return data["Volume"].mean()
    except Exception as e:
        print(f"Erreur : {e}")
        return None

    

