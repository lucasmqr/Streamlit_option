import streamlit as st


import sys
import os

# Ajouter le dossier parent au path Python pour pouvoir importer utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import st_option as sto

st.set_page_config(page_title="Equity")

st.title("Equity")

# --- Entrée utilisateur pour le ticker ---
ticker = st.text_input("Entrez le ticker de l'action :", "AAPL").upper()


if ticker:
    # --- Partie générale ---
    st.header("Informations générales")

    st.markdown("---")

period = st.selectbox("Période :", ["1mo", "3mo", "6mo", "1y", "ytd", "max"], index=2)

if ticker:
    data = sto.get_stock_price_history(ticker, period)
    if not data.empty:
        st.line_chart(data)
    else:
        st.warning("Aucune donnée de prix disponible pour ce ticker.")
    col1, col2 = st.columns(2)

    with col1:
        nom = sto.get_name(ticker)
        if nom:
            st.markdown(f"**Nom de l’entreprise :** {nom} ({ticker})")

        market_cap= sto.get_market_cap(ticker)
        market_cap=int(market_cap/10**9)

        if market_cap:
            st.metric(label="Market Cap", value=f"{market_cap:} bn USD")

        sector=sto.get_sector(ticker)

        if sector :
            st.write(sector)

        website =sto.get_website(ticker)

        if website :
            st.write(website)    
            description=sto.get_description(ticker)

        if description :
            st.write(description)           



    with col2:

        date = sto.get_last_price(ticker)[1]
        prix = sto.get_last_price(ticker)[0]

        if prix:
            st.metric(label=f"Last Price as of {date}",value=f"{prix:.2f} USD")
            var = sto.variation_depuis_ouverture(ticker)
            st.metric(label=" Variation depuis ouverture", value=f"{var} %", delta=f"{var} %")

        ytd=sto.performance_ytd(ticker)

        if ytd: 
            st.metric(label="YtD",value=f"{ytd} %", delta = f"{ytd} %")

        vol_actuel = sto.volume_depuis_ouverture(ticker)
        vol_moyen = sto.volume_moyen(ticker, "1mo")
        st.metric(label=" Volume échangé aujourd’hui", value=f"{vol_actuel:,}")
        st.metric(label=" Volume moyen (1 mois)", value=f"{vol_moyen:,}")

        beta=sto.get_beta(ticker)

        if beta :
            st.metric(label="Beta", value =f"{beta}")

        pe=sto.get_fwpe(ticker)
        if pe:
            st.metric(label="Forward PE", value=f"{pe:.2f}")






