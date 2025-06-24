"""

Commande à run dans le terminal : streamlit run "/Users/lucas/Documents Local/Code/Projet_Streamlit/st_option_front.py"

"""

import streamlit as st
import st_option as sto

st.title("Prix d'une action en temps réel")

ticker = st.text_input("Entrez le ticker de l'action :", "AAPL")

if ticker:
    prix = sto.last_price(ticker.upper())
    if prix is not None:
        st.success(f"Le dernier prix de {ticker.upper()} est : {prix:.2f} USD")
    else:
        st.error("Impossible de récupérer les données.")
    
periode = st.selectbox("Choisissez la période :", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd", "max"])

if ticker and periode:
    volume = sto.volume_total(ticker, periode)
    if volume:
        st.write(f"Volume total échangé pour {ticker.upper()} sur {periode} : {int(volume):,}")

